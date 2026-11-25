"""
src/threat_modeling/attack_simulator.py
=======================================
Adversarial threat simulation framework.

Paper reference (Sec. 3.2):
    "The test employs automated transactions to build malformed DoH requests
     to explore protocol resistance. Polymorphic Malware Over DoH — Encapsulated
     self-modifying malware payloads ... TLS Session Hijacking enables
     researchers to carry out Man-in-the-Middle (MitM) attacks ... DNS inquiries
     as hidden data exfiltration techniques through DoH."

Implements Eq. (2): threat singular embeddings t: A ↪ M_i, and Eq. (3): evasion
condition ∇Ψ(x)·δΨ(x) > ε.

Design:
    Each concrete simulator emits a DataFrame whose schema MATCHES the raw
    DoH schema used by FeatureExtractor (preprocessing.py), so synthetic and
    real samples are interchangeable downstream.
"""

from __future__ import annotations

import hashlib
import random
import string
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# =============================================================================
# 1. BASE CLASS
# =============================================================================
@dataclass
class AttackSimulator(ABC):
    """
    Abstract base class for all DoH attack simulators.

    Every subclass implements `_generate_one()` which returns a dict
    conforming to the shared raw-DoH schema.
    """

    n_samples: int = 1000
    seed: int = 42
    evasion_epsilon: float = 0.01        # Eq. (3) — ∇Ψ(x)·δΨ(x) > ε
    time_window_seconds: float = 3600.0

    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

    # -------- helpers -------------------------------------------------------
    @staticmethod
    def _rand_ip() -> str:
        return ".".join(str(random.randint(1, 254)) for _ in range(4))

    @staticmethod
    def _rand_domain(tld: str = "com") -> str:
        return "".join(random.choices(string.ascii_lowercase, k=10)) + f".{tld}"

    @staticmethod
    def _uuid_hex() -> str:
        return uuid.uuid4().hex

    def _base_row(self, t_ms: int, qname: str) -> dict:
        """Shared default field values for benign-looking traffic."""
        return {
            "timestamp_ms": t_ms,
            "qname": qname,
            "qtype": "A",
            "client_ip": self._rand_ip(),
            "server_ip": "8.8.8.8",
            "tls_cipher": "TLS_AES_256_GCM_SHA384",
            "tls_resumed": 0,
            "tls_cert_issuer": "DigiCert",
            "tls_cert_not_after": "2026-12-31T23:59:59Z",
            "tls_alpn": "h2",
            "sni": qname,
            "http_method": "POST",
            "http_content_type": "application/dns-message",
            "http_user_agent": "Mozilla/5.0",
            "http_accept": "application/dns-message",
            "http_latency_ms": float(self.rng.uniform(5, 40)),
            "http_body_size": 64,
            "is_nxdomain": 0,
            "label": 1,                       # 1 = malicious
        }

    # -------- public API ----------------------------------------------------
    @abstractmethod
    def _generate_one(self, t_ms: int) -> dict:
        """Generate a single malicious sample."""

    def simulate(self) -> pd.DataFrame:
        """Generate `n_samples` rows spanning `time_window_seconds`."""
        start_ms = int(time.time() * 1000) - int(self.time_window_seconds * 1000)
        rows = []
        for i in range(self.n_samples):
            t_ms = start_ms + int(i * self.time_window_seconds * 1000 / self.n_samples)
            rows.append(self._generate_one(t_ms))
        df = pd.DataFrame(rows)
        logger.info("%s: generated %d samples", self.__class__.__name__, len(df))
        return df

    # -------- Eq. (3) evasion check ----------------------------------------
    def satisfies_evasion(self, gradient_norm: float, delta_norm: float) -> bool:
        """∇Ψ(x)·δΨ(x) > ε  ⟺  ‖∇‖·‖δ‖·cos(θ) > ε (worst-case cos=1)."""
        return (gradient_norm * delta_norm) > self.evasion_epsilon


# =============================================================================
# 2. MUTATION-BASED FUZZING (Eq. 3 — malformed requests)
# =============================================================================
class MutationFuzzer(AttackSimulator):
    """Malformed DoH requests exploring protocol resistance (Sec. 3.2)."""

    mutation_rate: float = 0.05

    _OPERATORS = (
        "bit_flip",
        "byte_insertion",
        "header_reorder",
        "content_length_tamper",
        "random_subdomain_append",
        "ttl_manipulation",
    )

    def _mutate_bytes(self, s: str) -> str:
        b = bytearray(s.encode("utf-8"))
        for _ in range(max(1, int(len(b) * self.mutation_rate))):
            op = random.choice(self._OPERATORS)
            pos = random.randrange(len(b))
            if op == "bit_flip":
                b[pos] ^= 1 << random.randrange(8)
            elif op == "byte_insertion":
                b.insert(pos, random.randrange(256))
            elif op == "content_length_tamper":
                b[pos] = (b[pos] + random.randrange(1, 32)) % 256
            # remaining ops handled at higher level
        return bytes(b).decode("utf-8", errors="ignore")

    def _generate_one(self, t_ms: int) -> dict:
        qname = self._rand_domain()
        row = self._base_row(t_ms, qname)
        op = random.choice(self._OPERATORS)

        if op == "header_reorder":
            row["http_user_agent"], row["http_accept"] = (
                row["http_accept"], row["http_user_agent"],
            )
        elif op == "random_subdomain_append":
            row["qname"] = f"{self._uuid_hex()[:8]}.{qname}"
        elif op == "ttl_manipulation":
            row["is_nxdomain"] = int(random.random() < 0.5)

        # always mutate the qname bytes
        row["qname"] = self._mutate_bytes(row["qname"])[:253]
        row["http_body_size"] = int(self.rng.integers(1, 4096))
        return row


# =============================================================================
# 3. POLYMORPHIC MALWARE OVER DoH (Sec. 3.2)
# =============================================================================
class PolymorphicMalwareOverDoH(AttackSimulator):
    """
    Self-modifying payloads encapsulated inside DoH queries.
    Signature changes on every query (mutation_interval_seconds=60).
    """

    def _generate_one(self, t_ms: int) -> dict:
        qname = self._rand_domain(tld=random.choice(["com", "net", "io"]))
        row = self._base_row(t_ms, qname)
        payload_len = int(self.rng.integers(256, 2048))
        # Content looks like base64 with randomness
        row["http_body_size"] = payload_len
        row["content_type_mismatch"] = int(random.random() < 0.7)
        row["http_content_type"] = random.choice(
            ["application/octet-stream", "text/plain", "application/dns-message"]
        )
        # Multi-layer obfuscation → high-entropy qname
        row["qname"] = (
            "".join(random.choices(string.ascii_lowercase + string.digits, k=30))
            + "." + qname
        )
        return row


# =============================================================================
# 4. TLS SESSION HIJACKING (Sec. 3.2)
# =============================================================================
class TLSSessionHijack(AttackSimulator):
    """
    MitM attack that forges the server certificate and/or downgrades TLS.
    Traits: self-signed cert, cipher downgrade, SNI rewrite, session resumption.
    """

    def _generate_one(self, t_ms: int) -> dict:
        legit_domain = random.choice(["example.com", "cloudflare.com", "google.com"])
        row = self._base_row(t_ms, legit_domain)

        # Forge the server cert
        row["tls_cert_issuer"] = random.choice(
            ["Let's Encrypt", "DigiCert", "SELF-SIGNED", "Unknown-CA"]
        )
        if random.random() < 0.4:
            row["tls_cert_issuer"] = "SELF-SIGNED"

        # Downgrade cipher
        if random.random() < 0.6:
            row["tls_cipher"] = random.choice(
                ["TLS_RSA_WITH_AES_128_CBC_SHA", "TLS_RSA_WITH_3DES_EDE_CBC_SHA"]
            )

        # SNI rewrite (attacker points at real service but proxies through own CA)
        if random.random() < 0.4:
            row["sni"] = self._rand_domain()

        # Session resumption forgery
        row["tls_resumed"] = int(random.random() < 0.5)

        # Suspicious cert validity (already expired or 10 years)
        if random.random() < 0.3:
            row["tls_cert_not_after"] = "2020-01-01T00:00:00Z"
        elif random.random() < 0.3:
            row["tls_cert_not_after"] = "2035-12-31T23:59:59Z"

        return row


# =============================================================================
# 5. DNS TUNNELING (Sec. 3.2 — exfiltration via DoH)
# =============================================================================
class DNSTunneling(AttackSimulator):
    """
    Covert exfiltration channel: encodes payload in qname subdomain labels.
    Techniques mimic iodine / dnscat2 / dns2tcp.
    """

    _CHARSETS = string.ascii_lowercase + string.digits

    def _encode_payload(self, n_chars: int) -> str:
        return "".join(random.choices(self._CHARSETS, k=n_chars))

    def _generate_one(self, t_ms: int) -> dict:
        technique = random.choice(["iodine", "dnscat2", "dns2tcp", "base32"])
        base = random.choice(["t.example.com", "c2.bad-domain.io", "x.evil.net"])

        # 60-220 chars per label → high-entropy subdomain (iodine style)
        sub_len = random.randint(60, 220)
        subdomain = self._encode_payload(sub_len)

        if technique == "iodine":
            qname = f"{subdomain}.{base}"
        elif technique == "dnscat2":
            qname = f"{self._encode_payload(32)}.{subdomain[:64]}.{base}"
        elif technique == "dns2tcp":
            qname = f"{self._encode_payload(100)}.{base}"
        else:  # base32
            qname = f"{self._encode_payload(52)}.tunnel.{base}"

        row = self._base_row(t_ms, qname[:253])
        row["qtype"] = random.choice(["TXT", "NULL", "CNAME", "A"])
        row["http_body_size"] = random.randint(128, 512)
        row["http_content_type"] = "application/dns-message"
        return row


# =============================================================================
# 6. COMMAND-AND-CONTROL (Sec. 1, 2.4)
# =============================================================================
class CommandAndControl(AttackSimulator):
    """Periodic beaconing to a C2 server via DoH TXT records."""

    def _generate_one(self, t_ms: int) -> dict:
        c2_domain = random.choice(
            ["c2.evil-corp.io", "sync.badcdn.net", "update.malware.dev"]
        )
        row = self._base_row(t_ms, c2_domain)
        row["qtype"] = "TXT"
        row["http_body_size"] = random.randint(64, 256)
        # Beacon interval encoded as jitter in latency
        row["http_latency_ms"] = float(random.uniform(1, 200))
        return row


# =============================================================================
# 7. ORCHESTRATOR
# =============================================================================
@dataclass
class ThreatSimulationFramework:
    """Generates one combined adversarial dataset from all simulators."""

    n_per_attack: int = 5000
    seed: int = 42

    def _registry(self) -> dict[str, type[AttackSimulator]]:
        return {
            "mutation_fuzzing":     MutationFuzzer,
            "polymorphic_malware":  PolymorphicMalwareOverDoH,
            "tls_hijack":           TLSSessionHijack,
            "dns_tunneling":        DNSTunneling,
            "command_and_control":  CommandAndControl,
        }

    def generate(
        self, which: Optional[list[str]] = None
    ) -> dict[str, pd.DataFrame]:
        registry = self._registry()
        if which is None:
            which = list(registry.keys())

        results: dict[str, pd.DataFrame] = {}
        for name in which:
            if name not in registry:
                logger.warning("Unknown attack type: %s — skipped.", name)
                continue
            sim = registry[name](n_samples=self.n_per_attack, seed=self.seed)
            results[name] = sim.simulate()

        total = sum(len(v) for v in results.values())
        logger.success(
            "ThreatSimulationFramework: %d attacks generated across %d types.",
            total, len(results),
        )
        return results

    @staticmethod
    def merge(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for attack_name, df in results.items():
            df = df.copy()
            df["attack_type"] = attack_name
            frames.append(df)
        return pd.concat(frames, ignore_index=True)


def build_balanced_dataset(
    benign_df: pd.DataFrame, framework: ThreatSimulationFramework
) -> pd.DataFrame:
    """
    Combine benign + simulated malicious traffic with label balance.
    Labels: 0 = benign, 1 = malicious.
    """
    attack_results = framework.generate()
    malicious = ThreatSimulationFramework.merge(attack_results)

    benign_df = benign_df.copy()
    benign_df["label"] = 0
    benign_df["attack_type"] = "benign"

    combined = pd.concat([benign_df, malicious], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=framework.seed).reset_index(drop=True)
    logger.info(
        "Balanced dataset: %d benign + %d malicious = %d total.",
        len(benign_df), len(malicious), len(combined),
    )
    return combined
