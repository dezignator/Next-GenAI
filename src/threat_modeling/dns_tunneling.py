import random
import string
import numpy as np
import pandas as pd
from loguru import logger

class DNSTunneling:
    """Simulates data exfiltration using DNS queries (iodine/dnscat2 style)."""
    def __init__(self, n_samples: int = 2000, seed: int = 42):
        self.n_samples = n_samples
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def _encode_payload(self, length: int) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def inject(self) -> pd.DataFrame:
        logger.info(f"Injecting {self.n_samples} DNS Tunneling attacks...")
        rows = []
        for _ in range(self.n_samples):
            # Extremely long subdomains are the hallmark of tunneling
            payload = self._encode_payload(random.randint(60, 200))
            rows.append({
                "timestamp_ms": int(1.7e12),
                "qname": f"{payload}.exfil.evil.com",
                "qtype": str(random.choice(["TXT", "NULL", "CNAME"])),
                "client_ip": "10.0.0.4",
                "server_ip": "1.1.1.1",
                "tls_cipher": "TLS_AES_256_GCM_SHA384",
                "tls_resumed": 1,
                "tls_cert_issuer": "DigiCert",
                "tls_cert_not_after": "2026-12-31T23:59:59Z",
                "tls_alpn": "h2",
                "sni": "exfil.evil.com",
                "http_method": "POST",
                "http_content_type": "application/dns-message",
                "http_user_agent": "Mozilla/5.0",
                "http_accept": "application/dns-message",
                "http_latency_ms": float(self.rng.normal(50, 10)),
                "http_body_size": 512,
                "is_nxdomain": 0,
                "label": 1,
                "attack_type": "dns_tunneling"
            })
        return pd.DataFrame(rows)
