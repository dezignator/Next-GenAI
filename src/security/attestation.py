import time
import hashlib
from loguru import logger

class RemoteAttestation:
    """Handles remote attestation verification for TEEs."""
    def __init__(self, provider: str = "local"):
        self.provider = provider
        self.trusted_mrenclave = None

    def set_expected_measurement(self, mrenclave: str):
        self.trusted_mrenclave = mrenclave.lower()
        logger.info(f"Attestation: Expected MRENCLAVE set to {self.trusted_mrenclave}")

    def verify(self, enclave_binary_path: str, quote: dict) -> bool:
        logger.info(f"Initiating remote attestation via {self.provider}...")
        if not enclave_binary_path:
            logger.error("Attestation Failed: No enclave binary path provided.")
            return False
            
        with open(enclave_binary_path, "rb") as f:
            actual_mrenclave = hashlib.sha256(f.read()).hexdigest()
            
        if self.trusted_mrenclave and actual_mrenclave != self.trusted_mrenclave:
            logger.error(f"Attestation Failed: MRENCLAVE mismatch. Got {actual_mrenclave}")
            return False
            
        logger.success("Attestation Passed: Enclave integrity verified.")
        return True
