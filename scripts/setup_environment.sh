#!/bin/bash
# ==============================================================================
# Next-GenAI: V2 Environment Setup Script
# Sets up Python venv, installs dependencies, and initializes Kafka via Docker.
# ==============================================================================

set -e # Exit on error

echo "🚀 Initializing Next-GenAI V2 Environment..."

# 1. Setup Python Virtual Environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 2. Install Dependencies
echo "📦 Installing requirements..."
pip install --upgrade pip
pip install -e ".[dev,streaming,xai]"

# 3. Start Kafka & Flink Infrastructure
echo "🐳 Starting Kafka/Flink Docker containers..."
docker-compose -f docker/docker-compose.yml up -d

echo "⏳ Waiting for Kafka to be ready..."
sleep 10

# 4. Generate Toy Dataset & Inject Attacks
echo "⚔️ Building dataset and injecting V2 attacks..."
python scripts/build_dataset.py

echo "✅ Setup Complete. Run 'source venv/bin/activate' to start."
