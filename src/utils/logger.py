"""
src/utils/logger.py
===================
Centralized logging configuration (Loguru).

Provides:
    - Unified log format matching `configs/config.yaml → logging.format`
    - File rotation & retention (100 MB / 30 days)
    - Console sink with color-coded levels
    - Per-module child loggers (e.g., ``get_logger("capsnet")``)
    - Structured JSON mode for CI/CD and SIEM ingestion

Used by every module in the repo:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Model initialized")

Paper reference: no direct equation; supports reproducibility (Sec. 3).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger as _logger


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    dir: str = "logs"
    rotation_mb: int = 100
    retention_days: int = 30
    console: bool = True
    json_mode: bool = False          # structured JSON sink (SIEM)
    enqueue: bool = True             # multiprocess-safe async writes
    backtrace: bool = True
    diagnose: bool = False           # disable in production (leaks locals)


# =============================================================================
# 2. SINGLETON STATE
# =============================================================================
_configured: bool = False
_config: Optional[LoggingConfig] = None


# =============================================================================
# 3. PUBLIC CONFIGURE
# =============================================================================
def configure_logging(
    cfg: Optional[LoggingConfig] = None,
    force_reconfigure: bool = False,
) -> None:
    """
    Configure Loguru with file + console sinks.

    Safe to call multiple times (idempotent unless force_reconfigure=True).
    """
    global _configured, _config
    if _configured and not force_reconfigure:
        return

    cfg = cfg or LoggingConfig()
    _config = cfg

    # Remove default handler
    _logger.remove()

    # ---- Console sink ----
    if cfg.console:
        _logger.add(
            sys.stderr,
            level=cfg.level,
            format=cfg.format,
            colorize=True,
            backtrace=cfg.backtrace,
            diagnose=cfg.diagnose,
            enqueue=cfg.enqueue,
        )

    # ---- File sinks ----
    log_dir = Path(cfg.dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1) Full log — rotated
    _logger.add(
        log_dir / "nextgenai_{time:YYYY-MM-DD}.log",
        level=cfg.level,
        format=cfg.format,
        rotation=f"{cfg.rotation_mb} MB",
        retention=f"{cfg.retention_days} days",
        compression="gz",
        backtrace=cfg.backtrace,
        diagnose=cfg.diagnose,
        enqueue=cfg.enqueue,
    )

    # 2) Error-only log — long retention
    _logger.add(
        log_dir / "errors.log",
        level="ERROR",
        format=cfg.format,
        rotation=f"{cfg.rotation_mb} MB",
        retention="180 days",
        compression="gz",
        backtrace=True,
        diagnose=False,
        enqueue=cfg.enqueue,
    )

    # 3) Structured JSON for SIEM / log aggregation
    if cfg.json_mode:
        def _json_sink(message) -> None:                       # noqa: ANN001
            rec = message.record
            payload = {
                "timestamp": rec["time"].isoformat(),
                "level": rec["level"].name,
                "module": rec["name"],
                "function": rec["function"],
                "line": rec["line"],
                "message": rec["message"],
                "extra": rec.get("extra", {}),
            }
            with open(log_dir / "nextgenai_structured.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

        _logger.add(
            _json_sink,
            level=cfg.level,
            format="{message}",
            enqueue=cfg.enqueue,
        )

    _configured = True
    _logger.debug(
        f"Logging configured: level={cfg.level} dir={cfg.dir} "
        f"rotation={cfg.rotation_mb}MB retention={cfg.retention_days}d"
    )


# =============================================================================
# 4. CHILD LOGGER FACTORY
# =============================================================================
def get_logger(name: str = "nextgenai"):
    """
    Return a child logger with the given module name.
    Auto-configures logging on first call if the user forgot to.
    """
    if not _configured:
        # Honour env var override (e.g., in Docker)
        level = os.environ.get("NEXTGENAI_LOG_LEVEL", "INFO")
        configure_logging(LoggingConfig(level=level))
    return _logger.bind(module=name)


# =============================================================================
# 5. SPECIALIZED HELPERS
# =============================================================================
def log_experiment_start(experiment_name: str, config: dict) -> None:
    """Log experiment header — used at the top of every training run."""
    _logger.info("=" * 78)
    _logger.info(f"🚀 NEXT-GENAI EXPERIMENT: {experiment_name}")
    _logger.info(f"   Config keys: {len(config)}  |  Time: {__import__('datetime').datetime.utcnow().isoformat()}Z")
    _logger.info("=" * 78)


def log_experiment_end(experiment_name: str, metrics: dict) -> None:
    """Log experiment footer — final metric summary."""
    _logger.info("=" * 78)
    _logger.info(f"🏁 EXPERIMENT COMPLETE: {experiment_name}")
    for k, v in metrics.items():
        if isinstance(v, float):
            _logger.info(f"   {k:<24} = {v:.6f}")
        else:
            _logger.info(f"   {k:<24} = {v}")
    _logger.info("=" * 78)


def log_attack_simulation_summary(attack_results: dict) -> None:
    """
    Log per-attack sample counts (Sec. 3.2 threat simulation).
    """
    _logger.info("─" * 60)
    _logger.info("⚔️  Threat Simulation Summary")
    total = 0
    for name, df in attack_results.items():
        n = len(df)
        total += n
        _logger.info(f"   {name:<24} → {n:>8,} samples")
    _logger.info(f"   {'TOTAL':<24} → {total:>8,} samples")
    _logger.info("─" * 60)


def log_metrics_table(metrics: dict, title: str = "Metrics") -> None:
    """Pretty-print a metrics dict as a table."""
    _logger.info(f"📊 {title}")
    _logger.info(f"   {'metric':<28} | {'value':>10}")
    _logger.info("   " + "-" * 42)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            _logger.info(f"   {k:<28} | {v:>10.4f}")
        else:
            _logger.info(f"   {k:<28} | {str(v):>10}")


# =============================================================================
# 6. SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    configure_logging(LoggingConfig(level="DEBUG", json_mode=True))
    log = get_logger("demo")
    log.debug("debug message")
    log.info("info message")
    log.success("success message")
    log.warning("warning message")
    log.error("error message")
    log_experiment_start("demo_run", {"lr": 1e-3, "epochs": 50})
    log_metrics_table({"accuracy": 0.991, "f1": 0.988, "auc": 0.992}, "Fig. 1 targets")
    log_experiment_end("demo_run", {"accuracy": 0.991})
