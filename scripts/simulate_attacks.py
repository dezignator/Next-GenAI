"""
scripts/simulate_attacks.py
===========================
Generate the full adversarial dataset from all 5 attack simulators.

Paper reference (Sec. 3.2):
    - Mutation-based fuzzing         (Eq. 3 — evasion condition ∇Ψ(x)·δΨ(x) > ε)
    - Polymorphic malware over DoH
    - TLS session hijacking
    - DNS tunneling
    - Command-and-control (C2)

Also validates the Eq. (3) evasion property on generated samples.

Console script: nextgenai-attacks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.threat_modeling.attack_simulator import (
    CommandAndControl,
    DNSTunneling,
    MutationFuzzer,
    PolymorphicMalwareOverDoH,
    TLSSessionHijack,
    ThreatSimulationFramework,
)
from src.utils.config_loader import load_config
from src.utils.logger import (
    configure_logging,
    get_logger,
    log_attack_simulation_summary,
)


_ATTACK_CLASSES = {
    "mutation_fuzzing":    MutationFuzzer,
    "polymorphic_malware": PolymorphicMalwareOverDoH,
    "tls_hijack":          TLSSessionHijack,
    "dns_tunneling":       DNSTunneling,
    "command_and_control": CommandAndControl,
}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nextgenai-attacks")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output", type=str, default="data/synthetic/adversarial_set.parquet")
    p.add_argument(
        "--attack-all", action="store_true",
        help="Run all 5 attack types (default: yes)",
    )
    p.add_argument(
        "--samples-per-attack", type=int, default=None,
        help="Override per-attack sample count",
    )
    p.add_argument(
        "--attack", nargs="*", choices=list(_ATTACK_CLASSES.keys()),
        help="Subset of attack types to run",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--verify-evasion", action="store_true",
        help="Run Eq. 3 evasion property check on generated samples",
    )
    return p.parse_args(argv)


# =============================================================================
# EVASION CHECK (Eq. 3)
# =============================================================================
def _verify_evasion_property(df: pd.DataFrame, epsilon: float) -> dict:
    """
    Eq. (3): ∇Ψ(x) · δΨ(x) > ε

    We approximate the gradient with the qname entropy gradient — attackers
    want HIGH entropy in the tunnel/malware domain while keeping the DoH
    envelope benign-looking.
    """
    from src.data_acquisition.preprocessing import str_entropy
    entropies = df["qname"].fillna("").apply(str_entropy).to_numpy()
    if len(entropies) == 0:
        return {"mean_entropy": 0.0, "frac_above_eps": 0.0, "pass": False}
    eps_frac = float((entropies > epsilon).mean())
    return {
        "mean_entropy": float(entropies.mean()),
        "std_entropy":  float(entropies.std()),
        "frac_above_eps": eps_frac,
        "epsilon": epsilon,
        "pass": eps_frac > 0.5,   # majority of attack samples should evade
    }


# =============================================================================
# MAIN
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    configure_logging()
    log = get_logger("simulate_attacks")

    n_per = args.samples_per_attack or 5000
    log.info(f"Generating adversarial set — {n_per} samples per attack type")

    which = args.attack if args.attack else None
    framework = ThreatSimulationFramework(n_per_attack=n_per, seed=args.seed)
    results = framework.generate(which=which)

    log_attack_simulation_summary(results)

    # Merge
    combined = framework.merge(results)
    if "attack_type" not in combined.columns:
        raise RuntimeError("Missing attack_type column after merge.")

    log.info(f"Total adversarial samples: {len(combined):,}")
    log.info(f"Attack type distribution:\n{combined['attack_type'].value_counts()}")

    # Evasion property (Eq. 3)
    if args.verify_evasion:
        eps = float(cfg.adversarial.evasion_threshold_epsilon)
        log.info(f"Verifying Eq. (3) evasion property (ε={eps}) …")
        for attack_name, df in results.items():
            verdict = _verify_evasion_property(df, eps)
            status = "✅" if verdict["pass"] else "❌"
            log.info(
                f"  {status} {attack_name:<22} "
                f"mean_entropy={verdict['mean_entropy']:.2f} "
                f"frac_above_ε={verdict['frac_above_eps']:.2%}"
            )

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    log.success(f"✓ Saved adversarial set → {out}  ({len(combined):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
