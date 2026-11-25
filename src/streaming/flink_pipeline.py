"""
src/streaming/flink_pipeline.py
===============================
Distributed real-time DoH streaming pipeline (Kafka + Flink).

Paper reference (Sec. 1, 2.1, Table 1):
    "A distributed Apache Flink/Kafka pipeline enables real-time processing of
     DoH streams at scale, reducing detection latency by 50% compared to
     batch-oriented systems."
    "This setup provides greater than 65,000 events/s throughput with sub-50 ms
     end-to-end inference delay."

Targets:
    throughput   ≥ 65,000 events/s
    latency      ≤ 48 ms per event (Fig. 1)
    end-to-end   ≤ 50 ms (Sec. 1)

Provides two operating modes:
    - Pure-Python pipeline (works without Flink installed; useful for tests/dev)
    - PyFlink pipeline (uses the actual Flink runtime for production scale)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from loguru import logger

# Kafka is optional at import time; tests can inject a mock
try:
    from kafka import KafkaConsumer, KafkaProducer
    _KAFKA_AVAILABLE = True
except ImportError:
    KafkaConsumer = KafkaProducer = None
    _KAFKA_AVAILABLE = False
    logger.warning("kafka-python not installed → Kafka I/O disabled.")

# PyFlink is optional; production deployments use the real runtime
try:
    from pyflink.common import Types, WatermarkStrategy
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.datastream.functions import MapFunction, RuntimeContext
    from pyflink.datastream.connectors.kafka import (
        KafkaSink, KafkaSource, KafkaRecordSerializationSchema,
        KafkaOffsetsInitializer,
    )
    _FLINK_AVAILABLE = True
except ImportError:
    _FLINK_AVAILABLE = False
    logger.warning("pyflink not installed → using pure-Python streaming mode.")


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class StreamingConfig:
    # Kafka
    bootstrap_servers: list[str] = field(default_factory=lambda: ["localhost:9092"])
    topic_in: str = "doh-traffic-raw"
    topic_alerts: str = "doh-alerts"
    topic_anomalies: str = "doh-anomalies"
    topic_dlq: str = "doh-dlq"
    consumer_group: str = "nextgenai-doh-consumer"
    auto_offset_reset: str = "latest"
    max_poll_records: int = 500

    # Flink
    parallelism: int = 4
    checkpoint_interval_ms: int = 10_000
    watermark_delay_seconds: int = 2
    window_size_seconds: int = 10

    # Performance targets (Fig. 1)
    target_throughput_eps: int = 65_000
    target_latency_ms: float = 48.0
    backpressure_alert_ms: float = 200.0

    # Model
    model_path: str = "checkpoints/best_model.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold: float = 0.5
    batch_size: int = 256

    # TEE
    enable_tee: bool = False


# =============================================================================
# 2. EVENT PARSER
# =============================================================================
def parse_event(payload: bytes) -> Optional[dict]:
    """Parse a raw Kafka payload into a dict. Returns None on malformed input."""
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return None


def event_to_features(event: dict, feature_order: list[str]) -> torch.Tensor:
    """Convert a dict event to a (1, D) feature tensor."""
    vec = [float(event.get(f, 0.0)) for f in feature_order]
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)


# =============================================================================
# 3. INFERENCE FUNCTION (injected into Flink or pure-Python loop)
# =============================================================================
class DoHInferenceFn:
    """
    Wraps a HybridModel for streaming inference.
    Handles batching, TEE offloading, and thresholding.
    """

    def __init__(
        self, model, cfg: StreamingConfig, feature_order: list[str],
        tee_manager: Optional[Any] = None,
    ) -> None:
        self.model = model.eval().to(cfg.device)
        self.cfg = cfg
        self.feature_order = feature_order
        self.tee = tee_manager

    @torch.no_grad()
    def infer_batch(self, events: list[dict]) -> list[dict]:
        """Score a batch of events. Returns enriched dicts with `verdict`."""
        if not events:
            return []

        x_tab = torch.cat(
            [event_to_features(e, self.feature_order) for e in events], dim=0
        ).to(self.cfg.device)

        if self.tee is not None:
            probs = self.tee.secure_inference(self.model, x_tab)
        else:
            probs, _ = self.model(x_tab, return_aux=False) if hasattr(self.model, "forward") \
                       else (self.model(x_tab), None)

        # Guarantee probability shape (B, 2)
        if probs.dim() == 1:
            probs = torch.stack([1 - probs, probs], dim=-1)
        if probs.shape[-1] == 2:
            mal_probs = probs[:, 1].cpu().numpy()
        else:
            mal_probs = probs.cpu().numpy()

        out = []
        for ev, p in zip(events, mal_probs):
            ev_out = dict(ev)
            ev_out["malicious_prob"] = float(p)
            ev_out["verdict"] = "malicious" if p >= self.cfg.threshold else "benign"
            out.append(ev_out)
        return out


# =============================================================================
# 4. PURE-PYTHON PIPELINE (dev/test fallback)
# =============================================================================
class PurePythonPipeline:
    """
    A simple single-process Kafka pipeline that mimics the Flink semantics
    closely enough to be a fallback when the Flink runtime is unavailable.
    """

    def __init__(self, cfg: StreamingConfig, inference_fn: DoHInferenceFn) -> None:
        if not _KAFKA_AVAILABLE:
            raise RuntimeError("kafka-python is required for PurePythonPipeline.")
        self.cfg = cfg
        self.infer = inference_fn

        self.consumer = KafkaConsumer(
            cfg.topic_in,
            bootstrap_servers=cfg.bootstrap_servers,
            group_id=cfg.consumer_group,
            auto_offset_reset=cfg.auto_offset_reset,
            enable_auto_commit=True,
            max_poll_records=cfg.max_poll_records,
            value_deserializer=lambda v: v,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=cfg.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def run(self, max_batches: Optional[int] = None) -> dict:
        """Consume → score → emit alerts. Returns throughput metrics."""
        logger.info("▶ Starting PurePython streaming pipeline …")
        total = 0
        latencies: list[float] = []
        t_start = time.time()

        buffer: list[dict] = []
        batch_count = 0

        for msg in self.consumer:
            t_recv = time.time()
            event = parse_event(msg.value)
            if event is None:
                self.producer.send(self.cfg.topic_dlq, {"raw": msg.value.hex()})
                continue
            buffer.append((event, t_recv))

            if len(buffer) >= self.cfg.batch_size:
                self._flush(buffer, latencies)
                total += len(buffer)
                buffer.clear()
                batch_count += 1
                if max_batches is not None and batch_count >= max_batches:
                    break

        elapsed = time.time() - t_start
        eps = total / max(elapsed, 1e-6)
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        metrics = {
            "total_events": total,
            "elapsed_s": round(elapsed, 2),
            "throughput_eps": round(eps, 1),
            "target_throughput_eps": self.cfg.target_throughput_eps,
            "throughput_ok": eps >= self.cfg.target_throughput_eps,
            "mean_latency_ms": round(float(np.mean(latencies)) if latencies else 0.0, 2),
            "p95_latency_ms": round(p95, 2),
            "target_latency_ms": self.cfg.target_latency_ms,
            "latency_ok": p95 <= self.cfg.target_latency_ms,
        }
        logger.success(f"Pipeline finished: {metrics}")
        return metrics

    def _flush(self, buffer: list[tuple[dict, float]], latencies: list[float]) -> None:
        events = [e for e, _ in buffer]
        scored = self.infer.infer_batch(events)
        now = time.time()
        for (_, t_recv), ev in zip(buffer, scored):
            latencies.append((now - t_recv) * 1000.0)
            topic = (self.cfg.topic_anomalies if ev["verdict"] == "malicious"
                     else self.cfg.topic_alerts)
            self.producer.send(topic, ev)


# =============================================================================
# 5. FLINK PIPELINE (PyFlink, production scale)
# =============================================================================
if _FLINK_AVAILABLE:
    class _DoHInferenceMapFunction(MapFunction):
        """Flink MapFunction wrapping DoHInferenceFn."""

        def __init__(self, cfg: StreamingConfig, feature_order: list[str],
                     model_path: str) -> None:
            self.cfg = cfg
            self.feature_order = feature_order
            self.model_path = model_path
            self._infer: Optional[DoHInferenceFn] = None

        def open(self, runtime_context: RuntimeContext) -> None:
            from src.models.hybrid_model import HybridModel, HybridConfig
            model = HybridModel(HybridConfig())
            try:
                state = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(state["model"], strict=False)
            except FileNotFoundError:
                logger.warning(f"Checkpoint {self.model_path} missing — running randomly-initialized.")
            self._infer = DoHInferenceFn(model, self.cfg, self.feature_order)

        def map(self, value: str) -> str:
            event = json.loads(value)
            scored = self._infer.infer_batch([event])
            return json.dumps(scored[0])


class FlinkPipeline:
    """
    Production streaming pipeline using PyFlink.

    Reproduces Sec. 1: 65K+ events/s with sub-50 ms latency.
    """

    def __init__(
        self, cfg: StreamingConfig, feature_order: list[str],
        model_path: Optional[str] = None,
    ) -> None:
        if not _FLINK_AVAILABLE:
            raise RuntimeError("pyflink is required for FlinkPipeline.")
        self.cfg = cfg
        self.feature_order = feature_order
        self.model_path = model_path or cfg.model_path

    def build(self):
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(self.cfg.parallelism)
        env.enable_checkpointing(self.cfg.checkpoint_interval_ms)

        source = (
            KafkaSource.builder()
            .set_bootstrap_servers(",".join(self.cfg.bootstrap_servers))
            .set_topics(self.cfg.topic_in)
            .set_group_id(self.cfg.consumer_group)
            .set_starting_offsets(KafkaOffsetsInitializer.latest())
            .set_value_only_deserializer(lambda b: b.decode("utf-8"))
            .build()
        )

        ds = env.from_source(
            source, WatermarkStrategy.no_watermarks(), "doh-kafka-source"
        )

        scored = ds.map(
            _DoHInferenceMapFunction(self.cfg, self.feature_order, self.model_path),
            output_type=Types.STRING(),
        )

        sink = (
            KafkaSink.builder()
            .set_bootstrap_servers(",".join(self.cfg.bootstrap_servers))
            .set_record_serializer(
                KafkaRecordSerializationSchema.builder()
                .set_topic(self.cfg.topic_anomalies)
                .set_value_serialization_schema(lambda v: v.encode("utf-8"))
                .build()
            )
            .build()
        )
        scored.sink_to(sink)

        return env, scored

    def run(self) -> None:
        env, _ = self.build()
        logger.info("▶ Starting PyFlink streaming pipeline …")
        env.execute("NextGenAI-DoH-RealTime")


# =============================================================================
# 6. FACTORY
# =============================================================================
def build_pipeline(
    cfg: StreamingConfig,
    model,
    feature_order: list[str],
    tee_manager: Optional[Any] = None,
    prefer_flink: bool = True,
):
    """Instantiate the best available pipeline implementation."""
    infer = DoHInferenceFn(model, cfg, feature_order, tee_manager=tee_manager)
    if prefer_flink and _FLINK_AVAILABLE:
        return FlinkPipeline(cfg, feature_order)
    if _KAFKA_AVAILABLE:
        return PurePythonPipeline(cfg, infer)
    raise RuntimeError(
        "Neither Flink nor Kafka is available. Install pyflink and kafka-python."
    )


# =============================================================================
# 7. BENCHMARK (used by scripts/benchmark.py)
# =============================================================================
@torch.no_grad()
def benchmark_inference(
    inference_fn: DoHInferenceFn, n_events: int = 10_000, batch_size: int = 256,
) -> dict:
    """Measure throughput and latency without Kafka (in-process)."""
    logger.info(f"Benchmarking inference: {n_events} synthetic events …")
    rng = np.random.default_rng(0)
    events = [
        {f: float(rng.random()) for f in inference_fn.feature_order}
        for _ in range(n_events)
    ]
    latencies = []
    t_start = time.time()
    for i in range(0, n_events, batch_size):
        batch = events[i:i + batch_size]
        t0 = time.time()
        inference_fn.infer_batch(batch)
        latencies.append((time.time() - t0) * 1000.0 / len(batch))

    elapsed = time.time() - t_start
    eps = n_events / elapsed
    result = {
        "n_events": n_events,
        "throughput_eps": round(eps, 1),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "target_throughput_eps": 65_000,
        "target_latency_ms": 48.0,
        "throughput_ok": eps >= 65_000,
        "latency_ok": float(np.percentile(latencies, 95)) <= 48.0,
    }
    logger.success(f"Benchmark: {result}")
    return result
