import random
import numpy as np
import pandas as pd
from loguru import logger

class TLSHijack:
    """Simulates Man-in-the-Middle TLS session hijacking."""
    def __init__(self, n_samples: int = 2000, seed: int = 42):
        self.n_samples = n_samples
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def inject(self) -> pd.DataFrame:
        logger.info(f"Injecting {self.n_samples} TLS Hijack attacks...")
        rows = []
        for _ in range(self.n_samples):
            rows.append({
                "timestamp_ms": int(1.7e12),
                "qname": "legitimate-bank.com",
                "qtype": "A",
                "client_ip": "10.0.0.3",
                "server_ip": "1.1.1.1",
                "tls_cipher": "TLS_RSA_WITH_3DES_EDE_CBC_SHA", # Weak cipher
                "tls_resumed": 0,
                "tls_cert_issuer": "SELF-SIGNED", # MitM cert
                "tls_cert_not_after": "2026-12-31T23:59:59Z",
                "tls_alpn": "h2",
                "sni": "mitm-proxy.local", # SNI mismatch
                "http_method": "POST",
                "http_content_type": "application/dns-message",
                "http_user_agent": "Mozilla/5.0",
                "http_accept": "application/dns-message",
                "http_latency_ms": float(self.rng.uniform(150, 300)), # High latency
                "http_body_size": 64,
                "is_nxdomain": 0,
                "label": 1,
                "attack_type": "tls_hijack"
            })
        return pd.DataFrame(rows)
