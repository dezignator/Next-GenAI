import random
import string
import numpy as np
import pandas as pd
from loguru import logger

class MutationFuzzer:
    """Deep mutation-based fuzzing of DoH protocol headers and payloads."""
    def __init__(self, n_samples: int = 2000, mutation_rate: float = 0.05, seed: int = 42):
        self.n_samples = n_samples
        self.mutation_rate = mutation_rate
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def inject(self) -> pd.DataFrame:
        logger.info(f"Injecting {self.n_samples} Mutation Fuzzing attacks...")
        rows = []
        for _ in range(self.n_samples):
            qname = "".join(random.choices(string.ascii_lowercase, k=15)) + ".test.com"
            # Malform qname by inserting random bytes
            malformed_qname = list(qname)
            for _ in range(int(len(malformed_qname) * self.mutation_rate)):
                pos = random.randrange(len(malformed_qname))
                malformed_qname[pos] = random.choice(string.printable)
            
            rows.append({
                "timestamp_ms": int(1.7e12),
                "qname": "".join(malformed_qname),
                "qtype": "A",
                "client_ip": "10.0.0.1",
                "server_ip": "1.1.1.1",
                "tls_cipher": "TLS_RSA_WITH_AES_128_CBC_SHA", # Downgraded
                "tls_resumed": 0,
                "tls_cert_issuer": "Unknown",
                "tls_cert_not_after": "2020-01-01T00:00:00Z", # Expired
                "tls_alpn": "http/1.1",
                "sni": "mismatch.local",
                "http_method": "GET",
                "http_content_type": "text/plain", # Mismatch
                "http_user_agent": "fuzzer_bot",
                "http_accept": "*/*",
                "http_latency_ms": 500.0,
                "http_body_size": random.randint(500, 2000),
                "is_nxdomain": 1,
                "label": 1,
                "attack_type": "mutation_fuzzing"
            })
        return pd.DataFrame(rows)
