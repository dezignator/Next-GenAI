"""
src/training/trainer.py
=======================
Full training & evaluation loop for the HybridModel.

Paper reference (Sec. 3, Sec. 3.5, Abstract):
    - Adversarial training via perturbation-aware samples (Sec. 3.2, 3.5)
    - Mixed precision (needed for real-time throughput, Table 1)
    - Early stopping, top-k checkpoints
    - Target metrics:
          CapsNet accuracy = 99.1%   (Fig. 1)
          GraphTF recall   = 98.8%   (Fig. 1)
          CSSL AUC         = 0.992   (Fig. 1)
          FPR              = 0.007   (Fig. 1)
          TPR              = 0.991   (Fig. 1)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.hybrid_model import HybridModel


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    grad_accumulation_steps: int = 1

    warmup_epochs: int = 3
    min_lr: float = 1e-6

    early_stopping_patience: int = 8
    monitor: str = "val_auc"
    mode: str = "max"

    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3

    adversarial_training_ratio: float = 0.30
    cssl_weight: float = 0.1
    ce_weight: float = 0.5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 20
    seed: int = 42


# =============================================================================
# 2. METRICS
# =============================================================================
def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Full classification metric set used in the paper (Fig. 1)."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":       float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"]  = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"]  = 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics["fpr"] = float(fp / max(fp + tn, 1))
        metrics["tpr"] = float(tp / max(tp + fn, 1))
    else:
        metrics["fpr"] = metrics["tpr"] = 0.0
    return metrics


# =============================================================================
# 3. TRAINER
# =============================================================================
class Trainer:
    def __init__(
        self,
        model: HybridModel,
        cfg: TrainerConfig = TrainerConfig(),
        attack_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)

        # Ensure deterministic
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Optimizer & scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(cfg.epochs - cfg.warmup_epochs, 1),
            eta_min=cfg.min_lr,
        )

        # AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

        # Adversarial attack generator (Sec. 3.2)
        self.attack_fn = attack_fn

        # Checkpoint dir
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Early-stopping bookkeeping
        self._best_metric = -float("inf") if cfg.mode == "max" else float("inf")
        self._no_improvement = 0
        self._top_k: list[tuple[float, str]] = []

    # -------------------------------------------------------------------------
    # Adversarial sample augmentation (Sec. 3.2, 3.5)
    # -------------------------------------------------------------------------
    def _mix_adversarial(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attack_fn is None or self.cfg.adversarial_training_ratio <= 0:
            return x, y
        n_adv = int(self.cfg.adversarial_training_ratio * x.size(0))
        if n_adv == 0:
            return x, y
        idx = torch.randperm(x.size(0))[:n_adv]
        with torch.no_grad():
            adv_x = self.attack_fn(x[idx])
        adv_y = 1 - y[idx]           # adversarial = flipped label
        x = torch.cat([x[: x.size(0) - n_adv], adv_x], dim=0)
        y = torch.cat([y[: x.size(0) - n_adv], adv_y], dim=0)
        return x, y

    # -------------------------------------------------------------------------
    # One epoch
    # -------------------------------------------------------------------------
    def _run_epoch(self, loader, train: bool = True) -> tuple[dict, dict]:
        self.model.train(train)
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_caps: list[np.ndarray] = []
        all_gt: list[np.ndarray] = []
        all_cssl: list[np.ndarray] = []

        total_loss = 0.0
        n_samples = 0
        t0 = time.time()

        for step, batch in enumerate(loader):
            # Flexible batch: (x_tab, x_graph, A, y) or (x_tab, y)
            if len(batch) == 4:
                x_tab, x_graph, A, y = batch
            else:
                x_tab, y = batch[0], batch[-1]
                x_graph, A = None, None

            x_tab = x_tab.to(self.device, non_blocking=True)
            y = y.long().to(self.device, non_blocking=True)
            if x_graph is not None:
                x_graph = x_graph.to(self.device, non_blocking=True)
            if A is not None:
                A = A.to(self.device, non_blocking=True)

            if train:
                x_tab, y = self._mix_adversarial(x_tab, y)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
                    probs, aux = self.model(x_tab, x_graph, A, return_aux=True)
                    loss, comps = self.model.compute_loss(
                        probs, aux, y,
                        cssl_weight=self.cfg.cssl_weight,
                        ce_weight=self.cfg.ce_weight,
                    )

            if train:
                self.scaler.scale(loss / self.cfg.grad_accumulation_steps).backward()
                if (step + 1) % self.cfg.grad_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss += comps["total"] * y.size(0)
            n_samples += y.size(0)

            all_probs.append(probs[:, 1].detach().cpu().float().numpy())
            all_labels.append(y.cpu().numpy())
            all_caps.append(aux["capsnet"][:, 1].detach().cpu().float().numpy())
            all_gt.append(aux["graph_tf"][:, 1].detach().cpu().float().numpy())
            all_cssl.append(aux["cssl"][:, 1].detach().cpu().float().numpy())

            if train and (step + 1) % self.cfg.log_interval == 0:
                logger.debug(
                    f"[{'train' if train else 'val'}] step {step + 1} "
                    f"loss={comps['total']:.4f} margin={comps['margin']:.4f} "
                    f"ce={comps['ce']:.4f}"
                )

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        fused = compute_metrics(y_true, y_prob)
        fused["loss"] = total_loss / max(n_samples, 1)
        fused["time_s"] = time.time() - t0

        # Per-model metrics (for Fig. 1 verification)
        per_model = {
            "capsnet":  compute_metrics(y_true, np.concatenate(all_caps)),
            "graph_tf": compute_metrics(y_true, np.concatenate(all_gt)),
            "cssl":     compute_metrics(y_true, np.concatenate(all_cssl)),
        }
        return fused, per_model

    # -------------------------------------------------------------------------
    # Early-stopping / checkpointing
    # -------------------------------------------------------------------------
    def _update_early_stopping(self, val_metrics: dict) -> bool:
        cur = val_metrics[self.cfg.monitor]
        improved = (
            cur > self._best_metric if self.cfg.mode == "max"
            else cur < self._best_metric
        )
        if improved:
            self._best_metric = cur
            self._no_improvement = 0
            self._save_checkpoint(cur, tag="best")
            return False

        self._no_improvement += 1
        if self._no_improvement >= self.cfg.early_stopping_patience:
            logger.warning(
                f"Early stopping triggered after {self._no_improvement} epochs."
            )
            return True
        return False

    def _save_checkpoint(self, score: float, tag: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / f"{tag}_model.pt"
        torch.save(
            {"model": self.model.state_dict(), "score": score, "cfg": self.cfg.__dict__},
            path,
        )
        logger.info(f"💾 Saved checkpoint → {path}  ({self.cfg.monitor}={score:.4f})")

        # Top-k pruning
        self._top_k.append((score, str(path)))
        self._top_k.sort(key=lambda t: t[0], reverse=(self.cfg.mode == "max"))
        while len(self._top_k) > self.cfg.save_top_k:
            _, old_path = self._top_k.pop()
            try:
                os.remove(old_path)
            except OSError:
                pass

    # -------------------------------------------------------------------------
    # Public fit
    # -------------------------------------------------------------------------
    def fit(
        self,
        train_loader,
        val_loader,
        report_path: str = "results/training_report.json",
    ) -> dict:
        history: list[dict] = []
        report = {"epochs": [], "targets": {
            "capsnet_accuracy": 0.991,
            "graph_tf_recall":  0.988,
            "cssl_auc":         0.992,
            "fpr":              0.007,
            "tpr":              0.991,
        }}

        for epoch in range(1, self.cfg.epochs + 1):
            logger.info(f"═══ Epoch {epoch}/{self.cfg.epochs} ═══")

            tr_metrics, tr_per = self._run_epoch(train_loader, train=True)
            val_metrics, val_per = self._run_epoch(val_loader, train=False)

            # LR scheduling
            if epoch > self.cfg.warmup_epochs:
                self.scheduler.step()

            logger.info(
                f"TRAIN  | loss={tr_metrics['loss']:.4f}  "
                f"acc={tr_metrics['accuracy']:.4f}  f1={tr_metrics['f1']:.4f}  "
                f"auc={tr_metrics['roc_auc']:.4f}"
            )
            logger.info(
                f"VAL    | loss={val_metrics['loss']:.4f}  "
                f"acc={val_metrics['accuracy']:.4f}  f1={val_metrics['f1']:.4f}  "
                f"auc={val_metrics['roc_auc']:.4f}  fpr={val_metrics['fpr']:.4f}  "
                f"tpr={val_metrics['tpr']:.4f}"
            )
            logger.info(
                f"SUB    | capsnet_acc={tr_per['capsnet']['accuracy']:.4f}  "
                f"graph_recall={tr_per['graph_tf']['recall']:.4f}  "
                f"cssl_auc={tr_per['cssl']['roc_auc']:.4f}"
            )

            history.append({
                "epoch": epoch,
                "train": tr_metrics, "val": val_metrics,
                "train_per_model": tr_per, "val_per_model": val_per,
            })

            # Early stopping
            if self._update_early_stopping(val_metrics):
                break

        report["epochs"] = history
        report["best"] = {
            "monitor": self.cfg.monitor,
            "value": self._best_metric,
            "checkpoint": str(Path(self.cfg.checkpoint_dir) / "best_model.pt"),
        }

        # Save report
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.success(f"Training report → {report_path}")

        return report


# =============================================================================
# 4. STANDALONE EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate_model(
    model: HybridModel, test_loader, device: str = "cuda",
) -> dict:
    """Evaluate on test set and compare against Fig. 1 targets."""
    trainer = Trainer(model, TrainerConfig(device=device))
    metrics, per_model = trainer._run_epoch(test_loader, train=False)

    targets = {
        "capsnet_accuracy": 0.991,
        "graph_tf_recall":  0.988,
        "cssl_auc":         0.992,
        "fpr":              0.007,
        "tpr":              0.991,
    }
    diff = {
        "capsnet":    metrics["accuracy"] - targets["capsnet_accuracy"],
        "graph_tf":   per_model["graph_tf"]["recall"] - targets["graph_tf_recall"],
        "cssl":       per_model["cssl"]["roc_auc"] - targets["cssl_auc"],
        "fpr":        metrics["fpr"] - targets["fpr"],
        "tpr":        metrics["tpr"] - targets["tpr"],
    }
    logger.info(f"Fused : {metrics}")
    logger.info(f"Targets: {targets}")
    logger.info(f"Delta : {diff}")
    return {"fused": metrics, "per_model": per_model, "targets": targets, "delta": diff}
