"""
src/utils/metrics.py
====================
Metric aggregation, confusion-matrix helpers, and paper-target comparison.

Paper reference (Fig. 1, Table 1, Abstract):
    CapsNet Accuracy      = 99.1%
    Graph Transformer Recall = 98.8%
    CSSL AUC              = 0.992
    FPR                   = 0.007
    TPR                   = 0.991
    Throughput            = 65,000 events/s
    Latency               = 48 ms
    Adversarial Resilience gain = +80%

This module provides:
    - compute_all_metrics() — the exact 10-metric set from Sec. 3 evaluation
    - confusion_report() — per-class TP/FP/TN/FN
    - paper_target_diff() — compares a run's metrics against the paper
    - format_markdown_report() — for docs/results.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from loguru import logger


# =============================================================================
# 1. PAPER TARGETS (Fig. 1, Abstract, Table 1)
# =============================================================================
PAPER_TARGETS: dict[str, float] = {
    "capsnet_accuracy":       0.991,          # Abstract / Fig. 1
    "graph_tf_recall":        0.988,          # Abstract / Fig. 1
    "cssl_auc":               0.992,          # Fig. 1
    "fpr":                    0.007,          # Fig. 1
    "tpr":                    0.991,          # Fig. 1
    "throughput_eps":     65_000.0,           # Fig. 1
    "inference_latency_ms":  48.0,            # Fig. 1 / Table 1
    "adversarial_gain_pct":  80.0,            # Abstract ("80% improvement")
    "fpr_xai_pct":           0.001,           # Fig. 1 (XAI pipeline)
}


# =============================================================================
# 2. CORE METRIC COMPUTATION
# =============================================================================
def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute the full 10-metric set used in the paper.

    Parameters
    ----------
    y_true : (N,) int array with 0 (benign) / 1 (malicious)
    y_prob : (N,) float array with predicted P(malicious)
    threshold : classification cutoff (default 0.5 per config.yaml)

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc, pr_auc,
        fpr, tpr, balanced_accuracy, mcc
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
    }

    # AUC — requires both classes present
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    # FPR / TPR via confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics["fpr"] = float(fp / max(fp + tn, 1))
        metrics["tpr"] = float(tp / max(tp + fn, 1))
    else:
        metrics["fpr"] = 0.0
        metrics["tpr"] = 0.0

    return metrics


# =============================================================================
# 3. CONFUSION MATRIX REPORT
# =============================================================================
def confusion_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Return TP/TN/FP/FN and per-class stats (precision, recall, f1, support).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = map(int, cm.ravel())
    else:
        tn = fp = fn = tp = 0

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "per_class": {
            "benign": {
                "precision": float(prec[0]), "recall": float(rec[0]),
                "f1": float(f1[0]), "support": int(support[0]),
            },
            "malicious": {
                "precision": float(prec[1]), "recall": float(rec[1]),
                "f1": float(f1[1]), "support": int(support[1]),
            },
        },
        "support_total": int(support.sum()),
    }


# =============================================================================
# 4. PAPER-TARGET COMPARISON
# =============================================================================
def paper_target_diff(
    achieved: dict[str, float],
    tolerance: float = 0.005,
) -> dict:
    """
    Compare achieved metrics against PAPER_TARGETS.

    Parameters
    ----------
    achieved : e.g. {
        "capsnet_accuracy": 0.9905,
        "graph_tf_recall":  0.9870,
        "cssl_auc":         0.9915,
    }
    tolerance : acceptable |Δ| before flagging "FAIL"

    Returns
    -------
    dict with keys: matches, mismatches, summary
    """
    matches: dict[str, dict] = {}
    mismatches: dict[str, dict] = {}

    for metric_name, achieved_val in achieved.items():
        target_key = metric_name if metric_name in PAPER_TARGETS else None
        if target_key is None:
            logger.debug(f"No paper target for '{metric_name}' — skipping.")
            continue

        target_val = PAPER_TARGETS[target_key]
        delta = achieved_val - target_val

        # Relative comparisons for large-magnitude metrics (throughput)
        if target_key == "throughput_eps":
            relative = (achieved_val - target_val) / target_val
            ok = relative >= -tolerance
            delta_str = f"{delta:+.0f} eps ({relative:+.2%})"
        elif target_key == "inference_latency_ms":
            # Lower is better
            ok = achieved_val <= target_val * (1 + tolerance)
            delta_str = f"{delta:+.1f} ms"
        else:
            ok = abs(delta) <= tolerance
            delta_str = f"{delta:+.6f}"

        entry = {
            "achieved": achieved_val,
            "paper":    target_val,
            "delta":    delta_str,
            "pass":     bool(ok),
        }
        (matches if ok else mismatches)[metric_name] = entry

    summary = {
        "n_compared":   len(matches) + len(mismatches),
        "n_pass":       len(matches),
        "n_fail":       len(mismatches),
        "all_pass":     len(mismatches) == 0,
        "tolerance":    tolerance,
    }
    return {"matches": matches, "mismatches": mismatches, "summary": summary}


# =============================================================================
# 5. ROC CURVE DATA (for figures)
# =============================================================================
def roc_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, np.ndarray]:
    """Return FPR/TPR/thresholds for plotting (Fig. 1 ROC)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thr}


# =============================================================================
# 6. THRESHOLD SWEEP (calibration)
# =============================================================================
def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> list[dict]:
    """
    Sweep thresholds 0.05 → 0.95 and report metrics.
    Useful for `docs/results.md` and selecting the production cutoff.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        m = compute_all_metrics(y_true, y_prob, threshold=float(t))
        rows.append({
            "threshold": float(t),
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "fpr":       m["fpr"],
            "tpr":       m["tpr"],
        })
    return rows


# =============================================================================
# 7. MARKDOWN REPORT WRITER
# =============================================================================
def format_markdown_report(
    fused_metrics: dict[str, float],
    per_model: Optional[dict[str, dict]] = None,
    target_diff: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Emit a Markdown block for docs/results.md.
    Optionally saves to `output_path`.
    """
    lines: list[str] = []
    lines.append("## Evaluation Results\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in sorted(fused_metrics.items()):
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    if per_model:
        lines.append("\n### Per-Model Breakdown\n")
        lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
        lines.append("|-------|----------|-----------|--------|-----|---------|")
        for model_name, m in per_model.items():
            lines.append(
                f"| {model_name} | "
                f"{m.get('accuracy', 0):.4f} | "
                f"{m.get('precision', 0):.4f} | "
                f"{m.get('recall', 0):.4f} | "
                f"{m.get('f1', 0):.4f} | "
                f"{m.get('roc_auc', 0):.4f} |"
            )

    if target_diff:
        lines.append("\n### Paper Target Verification\n")
        lines.append("| Metric | Achieved | Paper | Δ | Status |")
        lines.append("|--------|----------|-------|---|--------|")
        for name, entry in target_diff["matches"].items():
            lines.append(
                f"| {name} | {entry['achieved']:.4f} | "
                f"{entry['paper']:.4f} | {entry['delta']} | ✅ |"
            )
        for name, entry in target_diff["mismatches"].items():
            lines.append(
                f"| {name} | {entry['achieved']:.4f} | "
                f"{entry['paper']:.4f} | {entry['delta']} | ❌ |"
            )
        s = target_diff["summary"]
        lines.append(
            f"\n**Target summary:** {s['n_pass']}/{s['n_compared']} pass, "
            f"{s['n_fail']} fail (tolerance = ±{s['tolerance']})\n"
        )

    report = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info(f"Markdown report saved → {output_path}")
    return report


# =============================================================================
# 8. JSON SERIALIZER (numpy-safe)
# =============================================================================
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):                                    # noqa: ANN001
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def save_json(payload: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, cls=_NumpyEncoder, indent=2)
    logger.info(f"JSON saved → {path}")


# =============================================================================
# 9. SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    from src.utils.logger import configure_logging, get_logger
    configure_logging()
    log = get_logger("metrics_demo")

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 1000)
    y_prob = np.clip(y_true * 0.7 + rng.normal(0.2, 0.2, 1000), 0, 1)

    m = compute_all_metrics(y_true, y_prob)
    log.info(f"Fused metrics: {m}")

    diff = paper_target_diff({
        "capsnet_accuracy": 0.9905,
        "graph_tf_recall":  0.9870,
        "cssl_auc":         0.9915,
    })
    log.info(f"Target diff: {diff['summary']}")

    md = format_markdown_report(m, target_diff=diff)
    print("\n" + md[:600] + "…\n")
