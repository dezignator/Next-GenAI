"""
scripts/evaluate.py
===================
Reproduce every Fig. 1 / Table 1 metric from a trained checkpoint.

Produces:
    results/evaluation_report.json   — full metric breakdown
    results/paper_target_diff.json   — achieved vs. paper targets
    results/roc_curve.npz            — for Fig. 1 ROC
    results/confusion.txt            — TP/FP/TN/FN breakdown

Paper reference (Fig. 1, Abstract, Table 1):
    CapsNet accuracy    = 99.1%
    GraphTF recall      = 98.8%
    CSSL AUC            = 0.992
    FPR                 = 0.007
    TPR                 = 0.991
    Throughput          = 65,000 events/s
    Latency             = 48 ms
    Adversarial gain    = +80%

Console script: nextgenai-evaluate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.__init__ import __version__, __paper_targets__
from src.data_acquisition.preprocessing import (
    ALL_ENGINEERED_COLS,
    FeatureExtractor,
    PreprocessingPipeline,
    Reducer,
)
from src.models.hybrid_model import HybridConfig, HybridModel
from src.training.trainer import TrainerConfig, compute_metrics
from src.utils.config_loader import get_device, load_config
from src.utils.logger import (
    configure_logging,
    get_logger,
    log_metrics_table,
)
from src.utils.metrics import (
    confusion_report,
    format_markdown_report,
    paper_target_diff,
    roc_curve_data,
    save_json,
    threshold_sweep,
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nextgenai-evaluate")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best_model.pt from training",
    )
    p.add_argument(
        "--test-set", type=str, default=None,
        help="Path to test parquet (overrides config)",
    )
    p.add_argument("--report", type=str, default="results/evaluation_report.json")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    return p.parse_args(argv)


# =============================================================================
# LOAD TEST SET
# =============================================================================
def _load_test_set(cfg, path: Optional[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    log = get_logger("evaluate.data")
    raw_path = Path(path or f"{cfg.data.processed_dir}/doh_test.parquet")
    if not raw_path.exists():
        log.warning(f"{raw_path} missing → using synthetic 1000-row test split")
        rng = np.random.default_rng(42)
        X = rng.random((1000, 64)).astype(np.float32)
        y = rng.integers(0, 2, 1000).astype(np.int64)
        return X, y, [f"f{i}" for i in range(64)]

    df = pd.read_parquet(raw_path) if raw_path.suffix == ".parquet" \
        else pd.read_csv(raw_path)
    log.info(f"Loaded test set: {len(df):,} rows from {raw_path}")

    if "label" in df.columns:
        y = df.pop("label").astype(np.int64).to_numpy()
    else:
        raise ValueError("Test set has no 'label' column.")

    feature_cols = [c for c in df.columns if c in ALL_ENGINEERED_COLS]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    return X, y, feature_cols


# =============================================================================
# MODEL LOADER
# =============================================================================
def _load_model(cfg, checkpoint: str, input_dim: int, device: str) -> HybridModel:
    log = get_logger("evaluate.model")
    model = HybridModel(HybridConfig(
        capsnet=__import__("src.models.capsule_network", fromlist=["CapsNetConfig"]).CapsNetConfig(
            input_dim=input_dim,
        ),
        cssl=__import__("src.models.cssl", fromlist=["CSSLConfig"]).CSSLConfig(
            encoder_in_dim=input_dim,
        ),
    ))
    state = torch.load(checkpoint, map_location="cpu")
    if "model" in state:
        state = state["model"]
    try:
        model.load_state_dict(state, strict=True)
        log.info(f"Loaded checkpoint (strict): {checkpoint}")
    except RuntimeError as e:
        log.warning(f"Strict load failed ({e}); retrying strict=False")
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# =============================================================================
# MAIN
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    configure_logging()
    log = get_logger("evaluate")

    device = get_device(cfg)
    log.info(f"Next-GenAI v{__version__}  —  scripts/evaluate.py")

    X_test, y_test, feature_cols = _load_test_set(cfg, args.test_set)
    log.info(f"Test shape: X={X_test.shape}  y={y_test.shape}")

    model = _load_model(cfg, args.checkpoint, X_test.shape[1], device)

    # Build loader
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    # Inference
    log.info("Running inference …")
    fused_probs, caps_probs, gt_probs, cssl_probs = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            probs, aux = model(x, return_aux=True)
            fused_probs.append(probs[:, 1].cpu().numpy())
            caps_probs.append(aux["capsnet"][:, 1].cpu().numpy())
            gt_probs.append(aux["graph_tf"][:, 1].cpu().numpy())
            cssl_probs.append(aux["cssl"][:, 1].cpu().numpy())
    elapsed = time.time() - t0
    n = len(y_test)
    throughput = n / elapsed
    latency_ms = (elapsed / n) * 1000.0
    log.info(
        f"Throughput: {throughput:,.0f} events/s | "
        f"Latency: {latency_ms:.2f} ms/event "
        f"(target ≥65,000 eps, ≤48 ms)"
    )

    # Metrics
    fused = compute_metrics(y_test, np.concatenate(fused_probs), threshold=args.threshold)
    per_model = {
        "capsnet":  compute_metrics(y_test, np.concatenate(caps_probs), threshold=args.threshold),
        "graph_tf": compute_metrics(y_test, np.concatenate(gt_probs), threshold=args.threshold),
        "cssl":     compute_metrics(y_test, np.concatenate(cssl_probs), threshold=args.threshold),
    }
    fused["throughput_eps"] = throughput
    fused["latency_ms"] = latency_ms

    log_metrics_table(fused, "Fused")
    for k, v in per_model.items():
        log_metrics_table(v, k)

    # Paper target diff
    achieved = {
        "capsnet_accuracy": per_model["capsnet"]["accuracy"],
        "graph_tf_recall":  per_model["graph_tf"]["recall"],
        "cssl_auc":         per_model["cssl"]["roc_auc"],
        "fpr":              fused["fpr"],
        "tpr":              fused["tpr"],
        "throughput_eps":   throughput,
        "inference_latency_ms": latency_ms,
    }
    diff = paper_target_diff(achieved, tolerance=0.05)   # 5% tolerance for benchmark noise
    s = diff["summary"]
    log.info(
        f"🎯 Paper target verification: {s['n_pass']}/{s['n_compared']} pass"
    )
    for name, entry in diff["mismatches"].items():
        log.warning(f"  ✗ {name}: {entry['delta']}")

    # Confusion matrix
    y_pred = (np.concatenate(fused_probs) >= args.threshold).astype(int)
    conf = confusion_report(y_test, y_pred)
    log.info(f"Confusion: {conf['confusion_matrix']}")

    # ROC
    roc = roc_curve_data(y_test, np.concatenate(fused_probs))

    # Threshold sweep
    sweep = threshold_sweep(y_test, np.concatenate(fused_probs))

    # Write artifacts
    out_dir = Path(args.report).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({
        "version": __version__,
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "n_samples": n,
        "fused": fused,
        "per_model": per_model,
        "confusion": conf,
        "paper_targets": __paper_targets__,
        "target_diff": diff,
        "threshold_sweep": sweep,
    }, str(out_dir / "evaluation_report.json"))

    np.savez(
        out_dir / "roc_curve.npz",
        fpr=roc["fpr"], tpr=roc["tpr"], thresholds=roc["thresholds"],
    )
    (out_dir / "confusion.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in conf["confusion_matrix"].items()),
        encoding="utf-8",
    )
    format_markdown_report(
        fused, per_model=per_model, target_diff=diff,
        output_path=str(out_dir / "results.md"),
    )

    log.success(f"✓ Evaluation complete → {out_dir}")
    return 0 if s["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
