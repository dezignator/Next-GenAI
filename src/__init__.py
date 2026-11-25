"""
Next-GenAI
==========

Next-generation AI for advanced threat detection and security enhancement in
DNS over HTTPS.

Reference:
    Ali, B., & Chen, G. (2024). Next-generation AI for advanced threat
    detection and security enhancement in DNS over HTTPS.
    Journal of Network and Computer Applications, Elsevier.

Institution:
    State Key Laboratory of Novel Software Technology,
    Nanjing University, Nanjing, Jiangsu 210093, China.

Modules:
    src.data_acquisition   — feature engineering (Sec. 3.1)
    src.threat_modeling    — adversarial attack simulation (Sec. 3.2)
    src.models             — CapsNet, Graph Transformer, CSSL (Sec. 3.3–3.5)
    src.training           — training loop and evaluation (Sec. 3.6)
    src.streaming          — Kafka + Flink pipeline (Sec. 1, 2.1)
    src.security           — TEE: SGX / TrustZone (Sec. 2.3, 3.7)
    src.explainability     — SHAP + LIME (Sec. 3.6)
    src.utils              — logging, metrics, config
"""

# =============================================================================
# VERSION — single source of truth (read by setup.py)
# =============================================================================
__version__ = "0.1.0"
__author__ = "Basharat Ali, Guihai Chen"
__email__ = "basharat@smail.nju.edu.cn, gchen@nju.edu.cn"
__license__ = "MIT"
__institution__ = "State Key Laboratory of Novel Software Technology, Nanjing University"

# Paper metadata
__paper_title__ = (
    "Next-generation AI for advanced threat detection and security "
    "enhancement in DNS over HTTPS"
)
__paper_journal__ = "Journal of Network and Computer Applications (Elsevier)"

# Paper-reported targets (Fig. 1, Abstract, Table 1) — canonical reference
__paper_targets__ = {
    "capsnet_accuracy": 0.991,
    "graph_tf_recall": 0.988,
    "cssl_auc": 0.992,
    "fpr": 0.007,
    "tpr": 0.991,
    "throughput_events_per_sec": 65_000,
    "inference_latency_ms": 48.0,
    "adversarial_resilience_gain_pct": 80.0,
    "xai_fpr_pct": 0.001,
}

# =============================================================================
# PUBLIC API (kept minimal to avoid circular imports)
# =============================================================================
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__institution__",
    "__paper_title__",
    "__paper_journal__",
    "__paper_targets__",
]
