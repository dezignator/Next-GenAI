"""
scripts/build_dataset.py
========================
Bridges the Toy Dataset Generator and the independent V2 Attack Injectors 
with the FeatureExtractor to produce a ready-to-train Parquet dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data_acquisition.toy_dataset import ToyDoHDatasetGenerator
from src.data_acquisition.preprocessing import FeatureExtractor, ALL_ENGINEERED_COLS
from src.threat_modeling.mutation_fuzzer import MutationFuzzer
from src.threat_modeling.polymorphic_malware import PolymorphicMalware
from src.threat_modeling.tls_hijack import TLSHijack
from src.threat_modeling.dns_tunneling import DNSTunneling
from src.utils.logger import configure_logging, get_logger

def main():
    configure_logging()
    log = get_logger("build_dataset")
    log.info("=== Building V2 Dataset (Toy + Injected Attacks) ===")

    # 1. Generate Base
    dataset_gen = ToyDoHDatasetGenerator(n_benign=10000)

    # 2. Inject V2 Attacks
    attacks = [
        MutationFuzzer(n_samples=2500).inject(),
        PolymorphicMalware(n_samples=2500).inject(),
        TLSHijack(n_samples=2500).inject(),
        DNSTunneling(n_samples=2500).inject()
    ]

    # 3. Combine
    raw_df = dataset_gen.build_environment(attacks)
    
    # 4. Extract Features (Sec. 3.1)
    log.info("Extracting features (DNS, TLS, DoH, Temporal)...")
    extractor = FeatureExtractor()
    feature_df = extractor.transform(raw_df)
    
    # Keep raw metadata for inspection
    feature_df["attack_type"] = raw_df["attack_type"]
    feature_df["qname"] = raw_df["qname"]
    
    # 5. Save
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "doh_v2_simulated.parquet"
    
    feature_df.to_parquet(out_path, index=False)
    log.success(f"Dataset saved to {out_path} | Shape: {feature_df.shape}")
    log.info(f"Label distribution:\n{feature_df['label'].value_counts()}")
    log.info(f"Attack distribution:\n{feature_df['attack_type'].value_counts()}")

if __name__ == "__main__":
    main()
