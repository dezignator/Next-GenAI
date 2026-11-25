"""
src/utils/config_loader.py
==========================
YAML → OmegaConf loader with CLI override support.

Wraps `configs/config.yaml` so every script uses the same accessor:

    from src.utils.config_loader import load_config
    cfg = load_config("configs/config.yaml")
    print(cfg.training.epochs)           # dot access
    print(cfg.capsnet.margin_loss.m_plus) # 0.9

Supports CLI overrides: --set training.epochs=100 --set device=cpu
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf


# =============================================================================
# 1. MAIN LOADER
# =============================================================================
def load_config(
    path: str = "configs/config.yaml",
    overrides: Optional[list[str]] = None,
    merge_env: bool = True,
) -> DictConfig:
    """
    Load a YAML config with optional CLI-style overrides.

    Parameters
    ----------
    path       : path to YAML config file
    overrides  : list of strings like ["training.epochs=100", "device=cpu"]
    merge_env  : if True, honour NEXTGENAI_<KEY>=<value> env vars

    Returns
    -------
    DictConfig with dot access + type coercion.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Config file is empty: {p}")

    cfg = OmegaConf.create(raw)

    # ---- CLI overrides ----
    if overrides:
        for expr in overrides:
            if "=" not in expr:
                raise ValueError(
                    f"Bad override '{expr}' — must be key=value"
                )
            key, value = expr.split("=", 1)
            # Type coercion: try int → float → bool → str
            coerced = _coerce_value(value)
            OmegaConf.update(cfg, key.strip(), coerced, merge=True)
            logger.debug(f"Override: {key.strip()} = {coerced}")

    # ---- Env overrides (NEXTGENAI_TRAINING_EPOCHS → training.epochs) ----
    if merge_env:
        for env_key, env_val in os.environ.items():
            if not env_key.startswith("NEXTGENAI_"):
                continue
            dotted = env_key[len("NEXTGENAI_"):].lower().replace("__", ".")
            try:
                OmegaConf.update(cfg, dotted, _coerce_value(env_val), merge=True)
                logger.debug(f"Env override: {dotted} = {env_val}")
            except Exception as e:                              # noqa: BLE001
                logger.warning(f"Env override failed for {env_key}: {e}")

    # ---- Type & structure validation ----
    _validate_required_keys(cfg)
    return cfg


# =============================================================================
# 2. VALUE COERCION
# =============================================================================
def _coerce_value(raw: str) -> Any:
    """Convert a CLI string to the most specific Python type."""
    s = raw.strip()
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    if s.lower() in ("null", "none", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # Lists / dicts
    if s.startswith("[") and s.endswith("]"):
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError:
            pass
    if s.startswith("{") and s.endswith("}"):
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError:
            pass
    return s


# =============================================================================
# 3. VALIDATION
# =============================================================================
_REQUIRED_TOP_LEVEL = (
    "project", "data", "capsnet", "graph_transformer",
    "cssl", "training", "evaluation",
)


def _validate_required_keys(cfg: DictConfig) -> None:
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise KeyError(
            f"Config is missing required top-level sections: {missing}"
        )
    # Deep-check a few critical hyperparameters against the paper
    _assert_eq(cfg.capsnet.margin_loss.m_plus, 0.9, "Eq. 7 m⁺")
    _assert_eq(cfg.capsnet.margin_loss.m_minus, 0.1, "Eq. 7 m⁻")
    _assert_eq(cfg.capsnet.margin_loss.lambda_, 0.5, "Eq. 7 λ")
    _assert_eq(cfg.cssl.temperature, 0.07, "Eq. 15 τ")
    _assert_eq(cfg.adversarial.evasion_threshold_epsilon, 0.01, "Eq. 3 ε")
    _assert_eq(cfg.graph_transformer.sheaf.laplacian_reg, 0.1, "Eq. 14 λ")


def _assert_eq(actual: float, expected: float, label: str, tol: float = 1e-9) -> None:
    if abs(float(actual) - expected) > tol:
        raise ValueError(
            f"{label}: config={actual} ≠ paper={expected}. "
            f"Refusing to run with mismatched hyperparameter."
        )


# =============================================================================
# 4. SAVE BACK
# =============================================================================
def save_config(cfg: DictConfig, path: str) -> None:
    """Persist the (possibly-overridden) config for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    logger.info(f"Config snapshot saved → {path}")


# =============================================================================
# 5. QUICK ACCESSORS
# =============================================================================
def get_device(cfg: DictConfig) -> str:
    """Resolve 'cuda'/'cpu'/'mps' from config, validating availability."""
    import torch
    requested = str(cfg.project.device)
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable → falling back to CPU.")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but unavailable → falling back to CPU.")
        return "cpu"
    return requested


def get_targets(cfg: DictConfig) -> dict[str, float]:
    """Extract the paper target metrics (used by scripts/evaluate.py)."""
    return {
        "capsnet_accuracy": 0.991,
        "graph_tf_recall": 0.988,
        "cssl_auc": 0.992,
        "fpr": 0.007,
        "tpr": 0.991,
        "throughput_eps": float(cfg.streaming.targets.throughput_events_per_sec),
        "latency_ms": float(cfg.streaming.targets.max_inference_latency_ms),
    }


# =============================================================================
# 6. SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    from src.utils.logger import configure_logging, get_logger
    configure_logging()
    log = get_logger("config_loader")

    cfg = load_config(
        "configs/config.yaml",
        overrides=["training.epochs=2", "device=cpu", "cssl.temperature=0.07"],
    )
    log.info(f"project: {cfg.project.name} v{cfg.project.version}")
    log.info(f"capsnet m_plus: {cfg.capsnet.margin_loss.m_plus}")
    log.info(f"cssl temperature: {cfg.cssl.temperature}")
    log.info(f"device: {get_device(cfg)}")
    log.info(f"targets: {get_targets(cfg)}")
