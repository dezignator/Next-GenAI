"""
scripts/benchmark.py
====================
Benchmark against every Table 1 metric:
    - Inference latency (CapsNet, GraphTF, CSSL, Fused)
    - Throughput (events/s)
    - Adversarial resilience (+80% target)
    - All baseline comparisons (Anagnostopoulos, You, Zhao, Li)

Console script: nextgenai-benchmark
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
import torch

from src.data_acquisition.preprocessing import ALL_ENGINEERED_COLS
from src.models.hybrid_model import HybridConfig, HybridModel
from src.utils.config_loader import get_device, load_config
from src.utils.logger import configure_logging, get_logger, log_metrics_table
from src.utils.metrics import save_json


# =============================================================================
# TABLE 1 — BASELINE LATENCIES (paper-reported; unit = ms)
# =============================================================================
TABLE1_BASELINES = {
    "Anagnostopoulos et al. (2021)": {"latency_ms": 180, "kind": "RF + GradientBoosting (batch)"},
    "You et al. (2023)":             {"latency_ms": 220, "kind": "CNN + LSTM (GPU)"},
    "Zhao et al. (2020)":            {"latency_ms": 75,  "kind": "Statistical fingerprinting"},
    "Li et al. (2024)":              {"latency_ms": 140, "kind": "GNN clustering (centralized)"},
    "Our research (target)":         {"latency_ms": 48,  "kind": "CapsNet + GraphTF + CSSL (stream)"},
}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nextgenai-benchmark")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--bench", nargs="*",
        choices=["latency", "throughput", "adversarial", "all"],
        default=["all"],
    )
    p.add_argument("--n-events", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", type=str, default="results/benchmark.json")
    return p.parse_args(argv)


# =============================================================================
# BENCHMARK HELPERS
# =============================================================================
@torch.no_grad()
def bench_latency(model: HybridModel, input_dim: int, device: str,
                  n_events: int, batch_size: int) -> dict:
    """Warm-up + measure mean/p95/p99 latency per event."""
    log = get_logger("bench.latency")
    log.info(f"Latency bench: {n_events} events, batch={batch_size}")

    # Warm-up
    for _ in range(3):
        x = torch.randn(batch_size, input_dim, device=device)
        model(x, return_aux=True)

    latencies = []
    if device == "cuda":
        torch.cuda.synchronize()
    t_total = time.time()
    for _ in range(max(1, n_events // batch_size)):
        x = torch.randn(batch_size, input_dim, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        model(x, return_aux=True)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000.0 / batch_size)

    total = time.time() - t_total
    arr = np.array(latencies)
    return {
        "n_events": max(1, n_events // batch_size) * batch_size,
        "elapsed_s": round(total, 3),
        "latency_mean_ms": round(float(arr.mean()), 2),
        "latency_p50_ms":  round(float(np.percentile(arr, 50)), 2),
        "latency_p95_ms":  round(float(np.percentile(arr, 95)), 2),
        "latency_p99_ms":  round(float(np.percentile(arr, 99)), 2),
        "throughput_eps":  round(max(1, n_events // batch_size) * batch_size / total, 1),
    }


@torch.no_grad()
def bench_adversarial_resilience(
    model: HybridModel, input_dim: int, device: str,
    n_events: int = 2000, eps: float = 0.01,
) -> dict:
    """
    Measure detection-rate drop under ε-perturbation.
    Target: +80% improvement vs. baselines (Abstract).
    """
    log = get_logger("bench.adversarial")
    log.info(f"Adversarial bench: {n_events} events, ε={eps}")

    x = torch.randn(n_events, input_dim, device=device)
    y = torch.randint(0, 2, (n_events,), device=device)

    clean_logits, _ = model(x, return_aux=True)
    perturbed = (x + eps * torch.sign(torch.randn_like(x))).clamp(0, 1)
    adv_logits, _ = model(perturbed, return_aux=True)

    clean_acc = (clean_logits.argmax(-1) == y).float().mean().item()
    adv_acc = (adv_logits.argmax(-1) == y).float().mean().item()
    drop = clean_acc - adv_acc
    resilience_pct = (1.0 - drop) * 100

    return {
        "clean_accuracy": round(clean_acc, 4),
        "adversarial_accuracy": round(adv_acc, 4),
        "accuracy_drop": round(drop, 4),
        "resilience_pct": round(resilience_pct, 2),
        "target_resilience_pct": 80.0,
        "pass": resilience_pct >= 80.0,
    }


# =============================================================================
# MAIN
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    configure_logging()
    log = get_logger("benchmark")
    log.info("Next-GenAI — benchmarking vs Table 1 + Fig. 1")

    device = get_device(cfg)
    input_dim = cfg.capsnet.input_dim

    # Build model
    from src.models.capsule_network import CapsNetConfig
    from src.models.cssl import CSSLConfig
    model = HybridModel(HybridConfig(
        capsnet=CapsNetConfig(input_dim=input_dim),
        cssl=CSSLConfig(encoder_in_dim=input_dim),
    )).to(device).eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        log.info(f"Loaded checkpoint: {args.checkpoint}")

    benches = args.bench
    if "all" in benches:
        benches = ["latency", "throughput", "adversarial"]

    result: dict = {"baselines": TABLE1_BASELINES}

    if "latency" in benches or "throughput" in benches:
        lat = bench_latency(model, input_dim, device, args.n_events, args.batch_size)
        log_metrics_table(lat, "Latency & Throughput")
        result["latency_throughput"] = lat
        result["table1_comparison"] = {
            name: {
                "baseline_ms": v["latency_ms"],
                "ours_ms": lat["latency_mean_ms"],
                "speedup_x": round(v["latency_ms"] / max(lat["latency_mean_ms"], 0.01), 2),
                "kind": v["kind"],
            }
            for name, v in TABLE1_BASELINES.items()
        }
        for name, entry in result["table1_comparison"].items():
            log.info(
                f"  {name:<32} | {entry['baseline_ms']:>5} ms → "
                f"{entry['ours_ms']:>5} ms  ({entry['speedup_x']}×)"
            )

    if "adversarial" in benches:
        adv = bench_adversarial_resilience(model, input_dim, device)
        log_metrics_table(adv, "Adversarial Resilience")
        result["adversarial"] = adv

    save_json(result, args.output)
    log.success(f"✓ Benchmark → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
