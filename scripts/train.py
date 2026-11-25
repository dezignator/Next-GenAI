"""
scripts/train.py
================
CLI entry point for training the HybridModel.

Reproduces the full training loop from the paper:
    - Loads config (configs/config.yaml)
    - Preprocesses raw DoH data (Sec. 3.1)
    - Generates adversarial samples (Sec. 3.2)
    - Pretrains CSSL on unlabeled traffic (Sec. 3.5)
    - Trains HybridModel (CapsNet + GraphTF + CSSL) jointly
    - Logs to TensorBoard + JSON reports
    - Saves top-k checkpoints

Paper targets verified at end of training (Fig. 1):
    capsnet_accuracy = 99.1%
    graph_tf_recall  = 98.8%
    cssl_auc         = 0.992
    fpr              = 0.007
    tpr              = 0.991

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --set training.epochs=100
    python scripts/train.py --config configs/config.yaml --adversarial-training
    python scripts/train.py --config configs/config.yaml --no-cssl-pretrain

Registered console_script:
    nextgenai-train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Allow `python scripts/train.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.__init__ import __version__, __paper_targets__
from src.data_acquisition.preprocessing import (
    FeatureExtractor,
    PreprocessingPipeline,
    Reducer,
)
from src.models.capsule_network import CapsNetConfig
from src.models.cssl import CSSLConfig
from src.models.graph_transformer import (
    GraphTransformerConfig,
    build_knn_graph,
)
from src.models.hybrid_model import HybridConfig, HybridModel
from src.threat_modeling.attack_simulator import (
    ThreatSimulationFramework,
    build_balanced_dataset,
)
from src.training.trainer import Trainer, TrainerConfig, compute_metrics
from src.utils.config_loader import get_device, load_config, save_config
from src.utils.logger import (
    configure_logging,
    get_logger,
    log_experiment_end,
    log_experiment_start,
    log_metrics_table,
)
from src.utils.metrics import paper_target_diff, save_json


# =============================================================================
# ARGPARSE
# =============================================================================
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nextgenai-train",
        description="Train HybridModel (CapsNet + GraphTF + CSSL) on DoH data.",
    )
    p.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config",
    )
    p.add_argument(
        "--set", dest="overrides", nargs="*", default=[],
        help="Config overrides: key=value (e.g., training.epochs=100)",
    )
    p.add_argument(
        "--adversarial-training", action="store_true",
        help="Enable adversarial augmentation (Sec. 3.2)",
    )
    p.add_argument(
        "--no-cssl-pretrain", action="store_true",
        help="Skip CSSL pretraining phase (Sec. 3.5)",
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="Override raw data path (otherwise from config)",
    )
    p.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for reports",
    )
    return p.parse_args(argv)


# =============================================================================
# DATA PREPARATION
# =============================================================================
def _prepare_datasets(cfg, args) -> dict:
    """
    Full pipeline: raw → features → split → adversarial augmentation.
    Returns dict of tensors and metadata.
    """
    log = get_logger("train.data")

    raw_path = Path(
        args.data
        or f"{cfg.data.raw_dir}/{cfg.data.dataset_name}.parquet"
    )
    if not raw_path.exists():
        # Fallback: synthesize if user hasn't downloaded the dataset
        log.warning(
            f"Raw dataset {raw_path} not found → generating synthetic data for smoke test."
        )
        raw_df = _synthesize_raw_data(n=5000, seed=cfg.project.seed)
    else:
        raw_df = pd.read_parquet(raw_path) if raw_path.suffix == ".parquet" \
            else pd.read_csv(raw_path)
        log.info(f"Loaded {len(raw_df):,} raw rows from {raw_path}")

    # Sec. 3.1 preprocessing
    pipeline = PreprocessingPipeline(
        extractor=FeatureExtractor(window_size=100),
        reducer=Reducer(
            pca_components=cfg.data.reduction.pca.n_components,
            pca_whiten=cfg.data.reduction.pca.whiten,
        ),
    )
    out = pipeline.run(
        raw_df,
        train_ratio=cfg.data.split.train_ratio,
        val_ratio=cfg.data.split.val_ratio,
        test_ratio=cfg.data.split.test_ratio,
        smote=(cfg.data.imbalance.strategy == "smote"),
        kl_threshold=cfg.adversarial.distributional_bias.kl_divergence_threshold,
    )
    log.info(
        f"Preprocessed: train={len(out['y_train']):,}  "
        f"val={len(out['y_val']):,}  test={len(out['y_test']):,}"
    )

    # Optional adversarial training injection (Sec. 3.2)
    if args.adversarial_training or cfg.adversarial.enabled:
        log.info("Generating adversarial samples (Sec. 3.2) …")
        framework = ThreatSimulationFramework(
            n_per_attack=500, seed=cfg.project.seed,
        )
        attack_frames = framework.generate()
        # Convert attack frames through same preprocessing pipeline
        adv_df = ThreatSimulationFramework.merge(attack_frames)
        adv_features = pipeline.extractor.transform(adv_df)
        adv_label = adv_features.pop("label").to_numpy(dtype=np.int64)
        from src.data_acquisition.preprocessing import ALL_ENGINEERED_COLS
        adv_X = adv_features[ALL_ENGINEERED_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        adv_X = pipeline.reducer.transform(adv_X)
        # Merge into training set
        out["X_train"] = np.concatenate([out["X_train"], adv_X], axis=0)
        out["y_train"] = np.concatenate([out["y_train"], adv_label], axis=0)
        log.info(
            f"Adversarial augmented training: {len(out['y_train']):,} total samples"
        )

    return out


def _synthesize_raw_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic raw DoH DataFrame matching FeatureExtractor schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "timestamp_ms": int(1.7e12 + i * 100),
            "qname": f"d{int(rng.integers(0, 500))}.example.com",
            "qtype": "A",
            "client_ip": ".".join(str(int(rng.integers(1, 255))) for _ in range(4)),
            "server_ip": "8.8.8.8",
            "tls_cipher": "TLS_AES_256_GCM_SHA384",
            "tls_resumed": int(rng.random() < 0.3),
            "tls_cert_issuer": "DigiCert",
            "tls_cert_not_after": "2026-12-31T23:59:59Z",
            "tls_alpn": "h2",
            "sni": f"d{int(rng.integers(0, 500))}.example.com",
            "http_method": "POST",
            "http_content_type": "application/dns-message",
            "http_user_agent": "Mozilla/5.0",
            "http_accept": "application/dns-message",
            "http_latency_ms": float(rng.normal(20, 5)),
            "http_body_size": int(rng.integers(32, 256)),
            "is_nxdomain": int(rng.random() < 0.05),
            "label": int(rng.random() < 0.4),
        })
    return pd.DataFrame(rows)


# =============================================================================
# DATA LOADERS
# =============================================================================
def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    weighted: bool = False,
) -> DataLoader:
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(x_t, y_t)
    sampler = None
    if weighted and shuffle:
        class_counts = np.bincount(y, minlength=2).astype(np.float64)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(y),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,      # keep 0 for deterministic tests
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# =============================================================================
# MODEL BUILDER
# =============================================================================
def _build_model(cfg, input_dim: int) -> HybridModel:
    hcfg = HybridConfig(
        capsnet=CapsNetConfig(
            input_dim=input_dim,
            primary_num_caps=cfg.capsnet.primary_capsules.num_capsules,
            primary_caps_dim=cfg.capsnet.primary_capsules.capsule_dim,
            num_classes=cfg.capsnet.num_classes,
            class_caps_dim=cfg.capsnet.class_capsule_dim,
            num_iterations=cfg.capsnet.routing.num_iterations,
            beta=cfg.capsnet.routing.beta,
            temperature=cfg.capsnet.routing.temperature,
            dropout=cfg.capsnet.dropout,
        ),
        graph_transformer=GraphTransformerConfig(
            d_model=cfg.graph_transformer.model.d_model,
            num_heads=cfg.graph_transformer.model.num_heads,
            num_layers=cfg.graph_transformer.model.num_layers,
            dim_feedforward=cfg.graph_transformer.model.dim_feedforward,
            dropout=cfg.graph_transformer.model.dropout,
            sheaf_enabled=cfg.graph_transformer.sheaf.enabled,
            sheaf_reg=cfg.graph_transformer.sheaf.laplacian_reg,
        ),
        cssl=CSSLConfig(
            encoder_in_dim=input_dim,
            temperature=cfg.cssl.temperature,
            projection_dim=cfg.cssl.projection_dim,
            projection_hidden=cfg.cssl.projection_hidden_dim,
        ),
        strategy=cfg.ensemble.strategy,
        weights=tuple(cfg.ensemble.weights.values()),
        threshold=cfg.ensemble.threshold,
    )
    return HybridModel(hcfg)


# =============================================================================
# MAIN
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # 1) Config & logging
    cfg = load_config(args.config, overrides=args.overrides)
    configure_logging()
    log = get_logger("train")

    log.info(f"Next-GenAI v{__version__}  —  scripts/train.py")
    log_experiment_start(
        "hybrid_training",
        {
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.optimizer.lr,
            "strategy": cfg.ensemble.strategy,
            "adversarial": cfg.adversarial.enabled,
        },
    )

    # Save config snapshot for reproducibility
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(out_dir / "config_snapshot.yaml"))

    # 2) Datasets
    data = _prepare_datasets(cfg, args)
    input_dim = data["X_train"].shape[1]
    log.info(f"Feature dimension after PCA: {input_dim}")

    train_loader = _make_loader(
        data["X_train"], data["y_train"],
        batch_size=cfg.training.batch_size, shuffle=True, weighted=True,
    )
    val_loader = _make_loader(
        data["X_val"], data["y_val"],
        batch_size=cfg.training.batch_size, shuffle=False,
    )
    test_loader = _make_loader(
        data["X_test"], data["y_test"],
        batch_size=cfg.training.batch_size, shuffle=False,
    )

    # 3) Model
    device = get_device(cfg)
    log.info(f"Building HybridModel on {device} …")
    model = _build_model(cfg, input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    # 4) CSSL pretraining (Sec. 3.5)
    if not args.no_cssl_pretrain and cfg.cssl.enabled:
        log.info("Phase 1: CSSL pretraining on unlabeled data (Sec. 3.5) …")
        unlabeled_loader = _make_loader(
            data["X_train"], np.zeros_like(data["y_train"]),
            batch_size=cfg.cssl.pretrain.batch_size, shuffle=True,
        )
        opt = torch.optim.AdamW(
            model.cssl.parameters(),
            lr=cfg.cssl.pretrain.learning_rate,
            weight_decay=cfg.cssl.pretrain.weight_decay,
        )
        history = model.pretrain_cssl(
            unlabeled_loader, opt, epochs=min(cfg.cssl.pretrain.epochs, 5)
        )
        log.info(f"CSSL pretraining done; final loss = {history[-1]:.4f}")
    else:
        log.info("Skipping CSSL pretraining (flag or config).")

    # 5) Adversarial training hook (Sec. 3.2)
    attack_fn = None
    if args.adversarial_training:
        log.info("Adversarial perturbation function enabled for training.")
        def attack_fn(x: torch.Tensor) -> torch.Tensor:
            noise = torch.randn_like(x) * cfg.adversarial.evasion_threshold_epsilon
            return (x + noise).clamp(0, 1)
        log.info(f"  perturbation ε = {cfg.adversarial.evasion_threshold_epsilon}")

    # 6) Trainer
    tcfg = TrainerConfig(
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        mixed_precision=cfg.training.mixed_precision,
        early_stopping_patience=cfg.training.early_stopping.patience,
        monitor=cfg.training.checkpoint.monitor,
        checkpoint_dir=cfg.training.checkpoint.dir,
        save_top_k=cfg.training.checkpoint.save_top_k,
        adversarial_training_ratio=cfg.adversarial.adversarial_training_ratio,
        device=device,
        seed=cfg.project.seed,
    )
    trainer = Trainer(model, tcfg, attack_fn=attack_fn)

    log.info("Phase 2: Training HybridModel …")
    report_path = str(out_dir / "training_report.json")
    report = trainer.fit(train_loader, val_loader, report_path=report_path)

    # 7) Final evaluation on test set
    log.info("Phase 3: Test evaluation …")
    model.eval()
    all_probs, all_labels = [], []
    caps_probs, gt_probs, cssl_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            probs, aux = model(x, return_aux=True)
            all_probs.append(probs[:, 1].cpu().numpy())
            all_labels.append(y.numpy())
            caps_probs.append(aux["capsnet"][:, 1].cpu().numpy())
            gt_probs.append(aux["graph_tf"][:, 1].cpu().numpy())
            cssl_probs.append(aux["cssl"][:, 1].cpu().numpy())

    y_true = np.concatenate(all_labels)
    fused = compute_metrics(y_true, np.concatenate(all_probs))
    per_model = {
        "capsnet":  compute_metrics(y_true, np.concatenate(caps_probs)),
        "graph_tf": compute_metrics(y_true, np.concatenate(gt_probs)),
        "cssl":     compute_metrics(y_true, np.concatenate(cssl_probs)),
    }

    log_metrics_table(fused, "Fused Model — Test Set")
    for name, m in per_model.items():
        log_metrics_table(m, f"{name} — Test Set")

    # 8) Paper target verification (Fig. 1)
    diff = paper_target_diff({
        "capsnet_accuracy": per_model["capsnet"]["accuracy"],
        "graph_tf_recall":  per_model["graph_tf"]["recall"],
        "cssl_auc":         per_model["cssl"]["roc_auc"],
        "fpr":              fused["fpr"],
        "tpr":              fused["tpr"],
    })
    s = diff["summary"]
    log.info(
        f"🎯 Paper target verification: {s['n_pass']}/{s['n_compared']} pass "
        f"(tolerance ±{s['tolerance']})"
    )
    if not s["all_pass"]:
        log.warning(f"Failing metrics: {list(diff['mismatches'].keys())}")

    # 9) Persist artifacts
    final_report = {
        "version": __version__,
        "config": cfg,
        "training": report,
        "test_fused": fused,
        "test_per_model": per_model,
        "paper_targets": __paper_targets__,
        "target_diff": diff,
    }
    final_path = out_dir / "final_report.json"
    save_json(final_report, str(final_path))

    log_experiment_end("hybrid_training", {
        "test_accuracy": fused["accuracy"],
        "test_f1": fused["f1"],
        "test_auc": fused["roc_auc"],
        "test_fpr": fused["fpr"],
        "targets_pass": f"{s['n_pass']}/{s['n_compared']}",
        "checkpoint": report["best"]["checkpoint"],
        "final_report": str(final_path),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
