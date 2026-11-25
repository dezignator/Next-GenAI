"""
scripts/download_dataset.py
===========================
Fetch the DoH dataset referenced in the paper (Abstract, Sec. 3.1):
    "doh_real_world_2022"

This script does NOT redistribute the dataset. It downloads it from the
canonical source OR generates a schema-equivalent synthetic fallback when
network access is unavailable (so first-time users can still run training).

Usage:
    python scripts/download_dataset.py --dataset doh_real_world_2022
    python scripts/download_dataset.py --dataset doh_real_world_2022 --synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.utils.logger import configure_logging, get_logger


_DATASET_REGISTRY = {
    "doh_real_world_2022": {
        "url": "https://zenodo.org/records/XXXXXXX/doh_real_world_2022.parquet",
        "sha256": "PUT_SHA256_HERE_AFTER_FIRST_DOWNLOAD",
        "size_mb": 180,
    },
}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="doh_real_world_2022",
                   choices=list(_DATASET_REGISTRY.keys()))
    p.add_argument("--output-dir", type=str, default="data/raw")
    p.add_argument("--synthetic", action="store_true",
                   help="Force synthetic generation (skip download)")
    p.add_argument("--n-rows", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    log = get_logger("download")
    log.info(f"Downloading {url} → {dest} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    log.success(f"Download complete: {dest}")


def _synthesize(n: int, seed: int) -> pd.DataFrame:
    """Generate schema-equivalent synthetic DoH data (Sec. 3.1 schema)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        malicious = int(rng.random() < 0.4)
        dlen = int(rng.integers(8, 60)) if not malicious else int(rng.integers(20, 200))
        domain = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=dlen))
        rows.append({
            "timestamp_ms": int(1.7e12 + i * 100),
            "qname": f"{domain}.{'c2.evil.io' if malicious else 'example.com'}",
            "qtype": rng.choice(["A", "AAAA", "TXT", "NULL"] if malicious else ["A", "AAAA"]),
            "client_ip": ".".join(str(int(rng.integers(1, 255))) for _ in range(4)),
            "server_ip": "8.8.8.8",
            "tls_cipher": str(rng.choice([
                "TLS_AES_256_GCM_SHA384", "TLS_RSA_WITH_AES_128_CBC_SHA",
            ])),
            "tls_resumed": int(rng.random() < (0.6 if malicious else 0.2)),
            "tls_cert_issuer": str(rng.choice(
                ["DigiCert", "SELF-SIGNED", "Unknown-CA"] if malicious else ["DigiCert", "Let's Encrypt"]
            )),
            "tls_cert_not_after": "2026-12-31T23:59:59Z",
            "tls_alpn": str(rng.choice(["h2", "h3", "http/1.1"])),
            "sni": f"{domain}.example.com" if not malicious else f"x.{domain}",
            "http_method": str(rng.choice(["POST", "GET"])),
            "http_content_type": str(rng.choice(
                ["application/dns-message"] if not malicious
                else ["application/dns-message", "text/plain", "application/octet-stream"]
            )),
            "http_user_agent": str(rng.choice(["Mozilla/5.0", "python-requests/2.31"])),
            "http_accept": "application/dns-message",
            "http_latency_ms": float(rng.normal(25 if not malicious else 200, 5 if not malicious else 80)),
            "http_body_size": int(rng.integers(32, 256) if not malicious else rng.integers(256, 2048)),
            "is_nxdomain": int(rng.random() < (0.4 if malicious else 0.05)),
            "label": malicious,
        })
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    log = get_logger("download_dataset")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.parquet"

    if args.synthetic:
        log.info(f"Generating synthetic {args.dataset} ({args.n_rows:,} rows) …")
        df = _synthesize(args.n_rows, args.seed)
        df.to_parquet(out_path, index=False)
        log.success(f"✓ Synthetic dataset → {out_path}")
        return 0

    meta = _DATASET_REGISTRY[args.dataset]
    try:
        _download(meta["url"], out_path)
    except Exception as e:                                    # noqa: BLE001
        log.warning(f"Download failed ({e}) → generating synthetic fallback.")
        df = _synthesize(args.n_rows, args.seed)
        df.to_parquet(out_path, index=False)
        log.success(f"✓ Synthetic fallback → {out_path}")
        return 0

    digest = _sha256(out_path)
    log.info(f"SHA-256: {digest}")
    if meta["sha256"] != "PUT_SHA256_HERE_AFTER_FIRST_DOWNLOAD":
        if digest != meta["sha256"]:
            log.error("SHA-256 mismatch — aborting.")
            return 1
        log.success("SHA-256 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
