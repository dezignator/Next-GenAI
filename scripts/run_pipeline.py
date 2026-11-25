"""
scripts/run_pipeline.py
=======================
Launch the real-time DoH streaming pipeline (Kafka + Flink).

Paper reference (Sec. 1, 2.1, Table 1):
    "distributed Apache Flink/Kafka pipeline ... 65,000 events/s throughput
     with sub-50 ms end-to-end inference delay"

Modes:
    --mode benchmark   : measure throughput/latency in-process (no Kafka)
    --mode kafka       : pure-Python Kafka consumer
    --mode flink       : PyFlink production pipeline
    --mode auto        : pick Flink if available, else Kafka, else benchmark

Console script: nextgenai-pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.data_acquisition.preprocessing import ALL_ENGINEERED_COLS
from src.models.hybrid_model import HybridConfig, HybridModel
from src.security.tee_manager import TEEConfig, TEEManager
from src.streaming.flink_pipeline import (
    DoHInferenceFn,
    StreamingConfig,
    benchmark_inference,
    build_pipeline,
)
from src.utils.config_loader import get_device, load_config
from src.utils.logger import configure_logging, get_logger, log_metrics_table


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nextgenai-pipeline")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--mode", choices=["benchmark", "kafka", "flink", "auto"],
        default="auto",
    )
    p.add_argument("--n-events", type=int, default=10_000,
                   help="For benchmark mode: how many synthetic events to score")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--kafka-brokers", type=str, default=None)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Kafka mode: stop after N batches")
    p.add_argument("--tee-mode", choices=["sgx", "trustzone", "simulation", "off"],
                   default="off")
    return p.parse_args(argv)


def _build_model(cfg, checkpoint: Optional[str], device: str) -> HybridModel:
    log = get_logger("pipeline.model")
    input_dim = cfg.capsnet.input_dim
    model = HybridModel(HybridConfig(
        capsnet=__import__("src.models.capsule_network", fromlist=["CapsNetConfig"]).CapsNetConfig(
            input_dim=input_dim,
        ),
        cssl=__import__("src.models.cssl", fromlist=["CSSLConfig"]).CSSLConfig(
            encoder_in_dim=input_dim,
        ),
    )).to(device).eval()

    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("model", state)
        try:
            model.load_state_dict(state, strict=True)
            log.info(f"Loaded checkpoint: {checkpoint}")
        except RuntimeError:
            model.load_state_dict(state, strict=False)
            log.warning(f"Partial load: {checkpoint}")
    else:
        log.warning("No checkpoint — using random init (smoke test only).")
    return model


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    configure_logging()
    log = get_logger("run_pipeline")
    log.info("Next-GenAI streaming pipeline")

    device = get_device(cfg)

    # Streaming config from YAML
    s_cfg = StreamingConfig(
        bootstrap_servers=(
            [args.kafka_brokers] if args.kafka_brokers
            else cfg.streaming.kafka.bootstrap_servers
        ),
        topic_in=cfg.streaming.kafka.topics.input,
        topic_alerts=cfg.streaming.kafka.topics.alerts,
        topic_anomalies=cfg.streaming.kafka.topics.anomalies,
        topic_dlq=cfg.streaming.kafka.topics.dead_letter,
        consumer_group=cfg.streaming.kafka.consumer_group,
        parallelism=cfg.streaming.flink.parallelism,
        target_throughput_eps=cfg.streaming.targets.throughput_events_per_sec,
        target_latency_ms=cfg.streaming.targets.max_inference_latency_ms,
        model_path=cfg.streaming.model_path if hasattr(cfg.streaming, "model_path")
                   else "checkpoints/best_model.pt",
        device=device,
        batch_size=args.batch_size,
        threshold=cfg.ensemble.threshold,
    )

    # Build model
    model = _build_model(cfg, args.checkpoint, device)

    # Optional TEE
    tee = None
    if args.tee_mode != "off":
        log.info(f"Enabling TEE backend: {args.tee_mode}")
        tee = TEEManager(TEEConfig(
            backend=args.tee_mode,
            attestation_enabled=(args.tee_mode != "simulation"),
            debug_logging=False,
        ))
        tee.initialize()
        report = tee.remote_attest()
        log.info(f"Attestation: verified={report.get('verified')}")

    # Feature order for streaming conversion
    feature_order = list(ALL_ENGINEERED_COLS)
    # Extend if PCA added more dims
    if cfg.capsnet.input_dim > len(feature_order):
        feature_order += [f"pc{i:03d}" for i in range(cfg.capsnet.input_dim - len(feature_order))]

    if args.mode == "benchmark":
        infer_fn = DoHInferenceFn(model, s_cfg, feature_order, tee_manager=tee)
        metrics = benchmark_inference(
            infer_fn, n_events=args.n_events, batch_size=args.batch_size,
        )
        log_metrics_table(metrics, "Streaming Benchmark")
        return 0 if (metrics["throughput_ok"] and metrics["latency_ok"]) else 1

    # Kafka or Flink
    prefer_flink = args.mode in ("flink", "auto")
    try:
        pipeline = build_pipeline(
            s_cfg, model, feature_order,
            tee_manager=tee, prefer_flink=prefer_flink,
        )
    except RuntimeError as e:
        log.error(f"Cannot build pipeline: {e}")
        log.info("Falling back to benchmark mode.")
        infer_fn = DoHInferenceFn(model, s_cfg, feature_order, tee_manager=tee)
        metrics = benchmark_inference(infer_fn, n_events=args.n_events)
        log_metrics_table(metrics, "Benchmark (fallback)")
        return 0

    log.info(f"Pipeline: {type(pipeline).__name__}")
    if hasattr(pipeline, "run"):
        try:
            if isinstance(pipeline, type(pipeline).__class__) and "PurePython" in type(pipeline).__name__:
                metrics = pipeline.run(max_batches=args.max_batches)
                log_metrics_table(metrics, "Kafka Pipeline")
            else:
                pipeline.run()
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
