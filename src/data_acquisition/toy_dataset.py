"""
Generates a toy DoH dataset so you can run the entire pipeline immediately.
Includes benign traffic generation and a mechanism to inject malicious attacks.
"""
import numpy as np
import pandas as pd
import time
from loguru import logger

class ToyDoHDatasetGenerator:
    def __init__(self, n_benign: int = 10000, seed: int = 42):
        self.n_benign = n_benign
        self.rng = np.random.default_rng(seed)

    def _generate_benign(self) -> pd.DataFrame:
        logger.info(f"Generating {self.n_benign} benign DoH samples...")
        rows = []
        start_time = int(time.time() * 1000) - 3600000 # 1 hr ago
        for i in range(self.n_benign):
            domain = "".join(self.rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=self.rng.integers(5, 15)))
            rows.append({
                "timestamp_ms": start_time + i * 100,
                "qname": f"{domain}.example.com",
                "qtype": str(self.rng.choice(["A", "AAAA"])),
                "client_ip": "192.168.1.1",
                "server_ip": "8.8.8.8",
                "tls_cipher": "TLS_AES_256_GCM_SHA384",
                "tls_resumed": int(self.rng.random() < 0.2),
                "tls_cert_issuer": "DigiCert",
                "tls_cert_not_after": "2026-12-31T23:59:59Z",
                "tls_alpn": "h2",
                "sni": f"{domain}.example.com",
                "http_method": "POST",
                "http_content_type": "application/dns-message",
                "http_user_agent": "Mozilla/5.0",
                "http_accept": "application/dns-message",
                "http_latency_ms": float(self.rng.normal(25, 5)),
                "http_body_size": int(self.rng.integers(32, 128)),
                "is_nxdomain": int(self.rng.random() < 0.05),
                "label": 0,
                "attack_type": "benign"
            })
        return pd.DataFrame(rows)

    def build_environment(self, attack_frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Combines benign with injected attacks."""
        benign = self._generate_benign()
        combined = pd.concat([benign] + attack_frames, ignore_index=True)
        combined = combined.sort_values("timestamp_ms").reset_index(drop=True)
        logger.success(f"Dataset built: {len(combined)} total samples ({combined['label'].sum()} malicious).")
        return combined
