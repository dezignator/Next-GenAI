"""
run_v2_master.py
================
The single entry-point to run the entire V2 pipeline:
1. Build toy dataset & inject attacks.
2. Preprocess & extract features.
3. Train HybridModel with Adversarial Training.
4. Evaluate against Table 1 targets.
"""
import subprocess
import sys
from pathlib import Path
from src.utils.logger import configure_logging, get_logger

def run_step(name: str, cmd: list[str]):
    log = get_logger("master_orchestrator")
    log.info(f"▶️ Executing: {name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"❌ {name} FAILED:\n{result.stderr}")
        sys.exit(1)
    log.success(f"✅ {name} COMPLETED.")

def main():
    configure_logging()
    log = get_logger("master")
    log.info("🚀 INITIALIZING NEXT-GENAI V2 MASTER PIPELINE 🚀")

    # Step 1: Build Dataset
    run_step("Dataset Generation & Attack Injection", 
             [sys.executable, "scripts/build_dataset.py"])

    # Step 2: Train
    log.info("Starting Model Training...")
    run_step("Model Training", 
             [sys.executable, "scripts/train.py", "--config", "configs/config.yaml", "--adversarial-training"])

    # Step 3: Evaluate
    log.info("Starting Evaluation...")
    run_step("Evaluation", 
             [sys.executable, "scripts/evaluate.py", "--checkpoint", "checkpoints/best_model.pt"])

    # Step 4: Benchmark
    log.info("Starting Benchmark...")
    run_step("Benchmarking", 
             [sys.executable, "scripts/benchmark.py", "--bench", "all"])

    log.info("🎉 ALL SYSTEMS GO. CHECK 'results/' FOR OUTPUTS.")

if __name__ == "__main__":
    main()
