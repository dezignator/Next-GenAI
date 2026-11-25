#!/bin/bash
# ==============================================================================
# Next-GenAI: V2 Full Benchmark Suite
# Reproduces Table 1 and Fig. 1 metrics.
# ==============================================================================

set -e

echo "📊 Running Next-GenAI V2 Benchmarks..."

echo "1️⃣ Testing Inference Latency (Target: 48ms)..."
python scripts/benchmark.py --bench latency --n-events 10000

echo "2️⃣ Testing Adversarial Resilience (Target: +80%)..."
python scripts/benchmark.py --bench adversarial

echo "3️⃣ Measuring Streaming Throughput (Target: 65,000 eps)..."
python scripts/run_pipeline.py --mode benchmark --n-events 50000

echo "✅ Benchmarks Complete. Results saved in results/benchmark.json"
