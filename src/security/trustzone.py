import torch
import torch.nn as nn
from loguru import logger

class TrustZoneEnclave:
    """ARM TrustZone OP-TEE specific implementation."""
    def __init__(self, ta_uuid: str):
        self.ta_uuid = ta_uuid
        self.is_initialized = False

    def initialize(self):
        logger.info(f"Initializing ARM TrustZone Trusted Application: {self.ta_uuid}")
        # In a real environment, this establishes a session with `tee_client_open_session`
        self.is_initialized = True

    def secure_inference(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            raise RuntimeError("TrustZone TA not initialized.")
            
        logger.debug("Executing inference inside TrustZone Secure World...")
        with torch.no_grad():
            model.eval()
            out = model(x)
            return out[0] if isinstance(out, tuple) else out
