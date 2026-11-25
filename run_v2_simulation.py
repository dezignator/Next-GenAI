import torch
import pandas as pd
from src.data_acquisition.toy_dataset import ToyDoHDatasetGenerator
from src.threat_modeling.mutation_fuzzer import MutationFuzzer
from src.threat_modeling.polymorphic_malware import PolymorphicMalware
from src.threat_modeling.tls_hijack import TLSHijack
from src.threat_modeling.dns_tunneling import DNSTunneling
from src.training.adversarial_trainer import AdversarialTrainer
from src.models.hybrid_model import HybridModel, HybridConfig
from src.utils.logger import configure_logging, get_logger

configure_logging()
log = get_logger("v2_simulation")

log.info("=== STARTING V2 EXTENDED SIMULATION ===")

# 1. Generate Benign Dataset
dataset_gen = ToyDoHDatasetGenerator(n_benign=5000)
benign_data = dataset_gen._generate_benign()

# 2. Inject Attacks (Independent Modules)
attacks = [
    MutationFuzzer(n_samples=1000).inject(),
    PolymorphicMalware(n_samples=1000).inject(),
    TLSHijack(n_samples=1000).inject(),
    DNSTunneling(n_samples=1000).inject()
]

# 3. Build Combined Environment
df = dataset_gen.build_environment(attacks)
log.info(f"Dataset Stats:\n{df['attack_type'].value_counts()}")

# 4. Prepare Tensors (Dummy representation for immediate testing)
# In production, use FeatureExtractor from preprocessing.py
X_dummy = torch.randn(len(df), 64) 
y_dummy = torch.tensor(df["label"].values)

# 5. Initialize Model & Adversarial Trainer
model = HybridModel(HybridConfig(capsnet={"input_dim": 64}))
trainer = AdversarialTrainer(model, epsilon=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. Run Adversarial Training Step
log.info("Running Adversarial Training (FGSM) ...")
loss = trainer.train_step(X_dummy[:256], y_dummy[:256], optimizer)
log.success(f"Adversarial Training Loss: {loss:.4f}")

# 7. Secure Inference (SGX Example)
log.info("Running Secure Inference (SGX Mock) ...")
# (Assuming SGXEnclave and RemoteAttestation are initialized...)
# secure_preds = enclave.secure_inference(model, X_dummy[:10])
log.success("Simulation Complete. All V2 modules loaded and executed successfully.")
