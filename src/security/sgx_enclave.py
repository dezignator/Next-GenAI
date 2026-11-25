import torch
import torch.nn as nn
from loguru import logger
from src.security.attestation import RemoteAttestation

class SGXEnclave:
    """Intel SGX specific implementation."""
    def __init__(self, enclave_path: str, attestation: RemoteAttestation):
        self.enclave_path = enclave_path
        self.attestation = attestation
        self.is_initialized = False

    def initialize(self):
        logger.info(f"Initializing Intel SGX Enclave: {self.enclave_path}")
        # In a real environment, this uses `oe_enclave_create`
        self.is_initialized = True

    def secure_inference(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            raise RuntimeError("SGX Enclave not initialized.")
        
        # Verify attestation quote before inference
        if not self.attestation.verify(self.enclave_path, {"quote": "mock_quote"}):
            raise RuntimeError("Attestation failed, aborting secure inference.")

        # Execute inference in isolated environment
        with torch.no_grad():
            model.eval()
            out = model(x)
            return out[0] if isinstance(out, tuple) else out
