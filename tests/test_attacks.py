"""
tests/test_attacks.py
=====================
Unit tests for attack simulators — verifies Eq. (2), (3), and Sec. 3.2 schema.
"""

import numpy as np
import pandas as pd
import pytest

from src.threat_modeling.attack_simulator import (
    CommandAndControl,
    DNSTunneling,
    MutationFuzzer,
    PolymorphicMalwareOverDoH,
    ThreatSimulationFramework,
    TLSSessionHijack,
    build_balanced_dataset,
)


# =============================================================================
# SCHEMA
# =============================================================================
_REQUIRED_COLS = {
    "timestamp_ms", "qname", "qtype", "client_ip", "server_ip",
    "tls_cipher", "tls_resumed", "tls_cert_issuer", "tls_cert_not_after",
    "tls_alpn", "sni", "http_method", "http_content_type",
    "http_user_agent", "http_accept", "http_latency_ms", "http_body_size",
    "is_nxdomain", "label",
}


@pytest.mark.parametrize("cls", [
    MutationFuzzer,
    PolymorphicMalwareOverDoH,
    TLSSessionHijack,
    DNSTunneling,
    CommandAndControl,
])
class TestAttackSchema:
    def test_output_has_required_columns(self, cls):
        sim = cls(n_samples=20, seed=0)
        df = sim.simulate()
        assert _REQUIRED_COLS.issubset(set(df.columns))

    def test_output_row_count(self, cls):
        sim = cls(n_samples=50, seed=0)
        df = sim.simulate()
        assert len(df) == 50

    def test_all_samples_malicious(self, cls):
        sim = cls(n_samples=30, seed=0)
        df = sim.simulate()
        assert (df["label"] == 1).all()

    def test_timestamps_increasing(self, cls):
        sim = cls(n_samples=20, seed=0)
        df = sim.simulate()
        assert df["timestamp_ms"].is_monotonic_increasing


# =============================================================================
# ATTACK-SPECIFIC BEHAVIOR
# =============================================================================
class TestMutationFuzzer:
    def test_produces_malformed_qnames(self):
        """Some qnames should contain non-ASCII bytes or very long labels."""
        sim = MutationFuzzer(n_samples=200, seed=0, mutation_rate=0.2)
        df = sim.simulate()
        # At least some mutated qnames differ from the base pattern
        valid_ratio = df["qname"].str.match(r"^[a-z]{10}\.(com|net|io)$").mean()
        assert valid_ratio < 1.0


class TestPolymorphicMalware:
    def test_qname_entropy_high(self):
        """Polymorphic malware embeds high-entropy payload in qname."""
        sim = PolymorphicMalwareOverDoH(n_samples=100, seed=0)
        df = sim.simulate()
        lengths = df["qname"].str.len()
        assert lengths.mean() > 25


class TestTLSSessionHijack:
    def test_self_signed_certificates_present(self):
        sim = TLSSessionHijack(n_samples=200, seed=0)
        df = sim.simulate()
        assert (df["tls_cert_issuer"] == "SELF-SIGNED").any()

    def test_session_resumption_present(self):
        sim = TLSSessionHijack(n_samples=200, seed=0)
        df = sim.simulate()
        assert (df["tls_resumed"] == 1).any()


class TestDNSTunneling:
    def test_qname_exfiltration_lengths(self):
        sim = DNSTunneling(n_samples=200, seed=0)
        df = sim.simulate()
        assert df["qname"].str.len().max() > 50

    def test_qtype_covert(self):
        """TXT / NULL / CNAME query types are classic tunneling."""
        sim = DNSTunneling(n_samples=500, seed=0)
        df = sim.simulate()
        covert = (df["qtype"].isin(["TXT", "NULL", "CNAME"])).mean()
        assert covert > 0.3


class TestCommandAndControl:
    def test_txt_records_used(self):
        sim = CommandAndControl(n_samples=100, seed=0)
        df = sim.simulate()
        assert (df["qtype"] == "TXT").all()


# =============================================================================
# Eq. (3) EVASION
# =============================================================================
class TestEvasionCondition:
    def test_evasion_when_gradient_exceeds_epsilon(self):
        sim = MutationFuzzer(n_samples=1, seed=0, evasion_epsilon=0.01)
        # Eq. (3): ‖∇‖·‖δ‖ > ε → True
        assert sim.satisfies_evasion(gradient_norm=1.0, delta_norm=1.0) is True

    def test_no_evasion_when_below_epsilon(self):
        sim = MutationFuzzer(n_samples=1, seed=0, evasion_epsilon=1.0)
        assert sim.satisfies_evasion(gradient_norm=0.01, delta_norm=0.01) is False


# =============================================================================
# ORCHESTRATOR
# =============================================================================
class TestThreatSimulationFramework:
    def test_generate_all_five(self):
        fw = ThreatSimulationFramework(n_per_attack=10, seed=0)
        results = fw.generate()
        assert set(results.keys()) == {
            "mutation_fuzzing", "polymorphic_malware", "tls_hijack",
            "dns_tunneling", "command_and_control",
        }
        assert all(len(df) == 10 for df in results.values())

    def test_generate_subset(self):
        fw = ThreatSimulationFramework(n_per_attack=5, seed=0)
        results = fw.generate(which=["mutation_fuzzing", "dns_tunneling"])
        assert set(results.keys()) == {"mutation_fuzzing", "dns_tunneling"}

    def test_merge_adds_attack_type_column(self):
        fw = ThreatSimulationFramework(n_per_attack=5, seed=0)
        results = fw.generate()
        combined = fw.merge(results)
        assert "attack_type" in combined.columns
        assert combined["attack_type"].nunique() == 5
        assert len(combined) == 25


class TestBalancedDatasetBuilder:
    def test_labels_balanced(self):
        benign = pd.DataFrame({c: ["x"] for c in _REQUIRED_COLS if c != "label"})
        benign = pd.concat([benign] * 20, ignore_index=True)
        for c in ["timestamp_ms", "tls_resumed", "http_latency_ms",
                  "http_body_size", "is_nxdomain"]:
            benign[c] = 0
        fw = ThreatSimulationFramework(n_per_attack=4, seed=0)
        combined = build_balanced_dataset(benign, fw)
        assert (combined["label"] == 0).sum() == 20
        assert (combined["label"] == 1).sum() == 20
        assert "attack_type" in combined.columns
