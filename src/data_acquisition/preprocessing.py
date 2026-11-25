"""
src/data_acquisition/preprocessing.py
======================================
Feature engineering pipeline for DoH traffic.

Paper reference (Sec. 3.1):
    "Ideal preprocessing accomplished feature engineering for relevant DoH
     traffic features. They were:
       - DNS Query Patterns — entropy, frequency, domain length distributions
       - TLS listening metadata — cipher suite usage, session resumption,
         certificates
       - DoH-Specific HTTP Header Analysis — content-type mismatch, HTTP
         request patterns, response latencies
       - Temporal Traffic Pattern — query timing, inter-arrival times
     The above-discussed features reduction methods, including PCA and t-SNE,
     were employed ..."

Also implements the KL-divergence dataset-bias test referenced in Table 1
(row 4: "KL-divergence test on distributional similarity").

CLI usage:
    python -m src.data_acquisition.preprocessing \\
        --input data/raw/doh_real_world_2022.csv \\
        --output data/processed/doh_processed.parquet
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# SMOTE is optional at runtime (see requirements.txt)
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SMOTE_AVAILABLE = False
    logger.warning("imbalanced-learn not installed → SMOTE disabled.")


# =============================================================================
# 1. CONSTANTS
# =============================================================================
DNS_FEATURE_COLS = [
    "query_entropy",
    "query_frequency",
    "domain_length",
    "subdomain_count",
    "tld_entropy",
    "nxdomain_rate",
]

TLS_FEATURE_COLS = [
    "cipher_suite_id",
    "session_resumption_rate",
    "cert_validity_days",
    "sni_mismatch",
    "alpn_h2",
    "alpn_h3",
    "alpn_http11",
]

DOH_FEATURE_COLS = [
    "content_type_mismatch",
    "request_method_post",
    "response_latency_ms",
    "user_agent_entropy",
    "accept_header_mismatch",
    "body_size_bytes",
]

TEMPORAL_FEATURE_COLS = [
    "inter_arrival_ms",
    "burstiness",
    "periodicity_score",
    "hour_of_day_entropy",
]

ALL_ENGINEERED_COLS = (
    DNS_FEATURE_COLS + TLS_FEATURE_COLS + DOH_FEATURE_COLS + TEMPORAL_FEATURE_COLS
)


# =============================================================================
# 2. LOW-LEVEL FEATURE HELPERS
# =============================================================================
def shannon_entropy(tokens: Iterable[str], base: float = 2.0) -> float:
    """Shannon entropy of a token sequence (used for query_entropy, tld_entropy)."""
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    _, counts = np.unique(tokens, return_counts=True)
    return float(scipy_entropy(counts, base=base))


def str_entropy(s: str, base: float = 2.0) -> float:
    """Character-level entropy of a single string (DNS qname, user-agent, etc.)."""
    if not s:
        return 0.0
    chars, counts = np.unique(list(s), return_counts=True)
    return float(scipy_entropy(counts, base=base))


def burstiness_score(timestamps_ms: np.ndarray) -> float:
    """
    Goh & Barabási burstiness B = (σ - μ) / (σ + μ), bounded in [-1, 1].
    B = -1 → perfectly regular; B = +1 → highly bursty.
    """
    if len(timestamps_ms) < 3:
        return 0.0
    iat = np.diff(np.sort(timestamps_ms))
    mu, sigma = iat.mean(), iat.std()
    if mu + sigma == 0:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


def periodicity_score(timestamps_ms: np.ndarray, fs_hz: float = 1.0) -> float:
    """
    Peak of normalized autocorrelation → higher = more periodic.
    Detects regular C2 beaconing.
    """
    if len(timestamps_ms) < 8:
        return 0.0
    iat = np.diff(np.sort(timestamps_ms)).astype(np.float64)
    iat = iat - iat.mean()
    if np.allclose(iat, 0.0):
        return 0.0
    ac = np.correlate(iat, iat, mode="full")[len(iat) - 1:]
    ac = ac / (ac[0] + 1e-12)
    return float(np.clip(ac[1:].max(), 0.0, 1.0))


# =============================================================================
# 3. FEATURE EXTRACTOR (4 families, Sec. 3.1)
# =============================================================================
@dataclass
class FeatureExtractor:
    """
    Stateless transformer: raw DoH flow DataFrame → engineered features.

    Expected raw columns (rename with --column-map if different):
        timestamp_ms, qname, qtype, client_ip, server_ip,
        tls_cipher, tls_resumed, tls_cert_issuer, tls_cert_not_after,
        tls_alpn, sni, http_method, http_content_type, http_user_agent,
        http_accept, http_latency_ms, http_body_size,
        is_nxdomain, label
    """

    window_size: int = 100
    enable_dns: bool = True
    enable_tls: bool = True
    enable_doh: bool = True
    enable_temporal: bool = True

    # -------------------------------------------------------------------------
    # 3.1 DNS Query Patterns (Sec. 3.1, bullet 1)
    # -------------------------------------------------------------------------
    def _dns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Extracting DNS query-pattern features …")
        out = pd.DataFrame(index=df.index)

        out["query_entropy"] = df["qname"].fillna("").apply(str_entropy)
        out["domain_length"] = df["qname"].fillna("").str.len()
        out["subdomain_count"] = (
            df["qname"].fillna("").str.count(r"\.") - 1
        ).clip(lower=0)
        out["tld_entropy"] = (
            df["qname"]
            .fillna("")
            .str.rsplit(".", n=1)
            .str[-1]
            .apply(str_entropy)
        )

        # Frequency of a qname within a sliding temporal window (rolling)
        qname_counts = df.groupby("qname")["timestamp_ms"].transform("count")
        out["query_frequency"] = qname_counts.astype(np.float32)

        # NXDOMAIN rate over window
        out["nxdomain_rate"] = (
            df["is_nxdomain"]
            .fillna(0)
            .astype(np.int8)
            .rolling(window=min(self.window_size, len(df)), min_periods=1)
            .mean()
        )

        return out

    # -------------------------------------------------------------------------
    # 3.2 TLS Listening Metadata (Sec. 3.1, bullet 2)
    # -------------------------------------------------------------------------
    def _tls_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Extracting TLS metadata features …")
        out = pd.DataFrame(index=df.index)

        # Cipher suite → stable integer id
        out["cipher_suite_id"] = df["tls_cipher"].astype("category").cat.codes

        # Session-resumption rate over window
        out["session_resumption_rate"] = (
            df["tls_resumed"]
            .fillna(0)
            .astype(np.int8)
            .rolling(window=min(self.window_size, len(df)), min_periods=1)
            .mean()
        )

        # Certificate validity in days (expiry - now)
        not_after = pd.to_datetime(df["tls_cert_not_after"], errors="coerce", utc=True)
        now = pd.Timestamp.utcnow()
        out["cert_validity_days"] = (not_after - now).dt.total_seconds() / 86400.0
        out["cert_validity_days"] = out["cert_validity_days"].fillna(-1).clip(-1, 3650)

        # SNI mismatch against qname
        def _sni_mismatch(row: pd.Series) -> int:
            sni = str(row.get("sni") or "")
            qname = str(row.get("qname") or "")
            if not sni or not qname:
                return 0
            return int(not qname.endswith(sni))

        out["sni_mismatch"] = df.apply(_sni_mismatch, axis=1).astype(np.int8)

        # ALPN one-hot
        alpn = df["tls_alpn"].fillna("").str.lower()
        out["alpn_h2"] = (alpn == "h2").astype(np.int8)
        out["alpn_h3"] = (alpn.isin(["h3", "h3-29", "h3-32"])).astype(np.int8)
        out["alpn_http11"] = (alpn == "http/1.1").astype(np.int8)

        return out

    # -------------------------------------------------------------------------
    # 3.3 DoH-Specific HTTP Header Analysis (Sec. 3.1, bullet 3)
    # -------------------------------------------------------------------------
    def _doh_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Extracting DoH HTTP-header features …")
        out = pd.DataFrame(index=df.index)

        # Content-type mismatch: DoH MUST be application/dns-message
        ct = df["http_content_type"].fillna("").str.lower()
        out["content_type_mismatch"] = (
            ~ct.str.contains("application/dns-message", regex=False)
        ).astype(np.int8)

        # GET vs POST (RFC 8484 permits both; attackers over-use one)
        out["request_method_post"] = (
            df["http_method"].fillna("").str.upper() == "POST"
        ).astype(np.int8)

        # Response latency
        out["response_latency_ms"] = (
            df["http_latency_ms"].fillna(0).astype(np.float32)
        )

        # User-agent entropy (empty UA = suspicious)
        out["user_agent_entropy"] = (
            df["http_user_agent"].fillna("").apply(str_entropy)
        )

        # Accept header mismatch (should be application/dns-message)
        accept = df["http_accept"].fillna("").str.lower()
        out["accept_header_mismatch"] = (
            ~accept.str.contains("application/dns-message", regex=False)
        ).astype(np.int8)

        # Body size distribution
        out["body_size_bytes"] = df["http_body_size"].fillna(0).astype(np.float32)

        return out

    # -------------------------------------------------------------------------
    # 3.4 Temporal Traffic Pattern (Sec. 3.1, bullet 4)
    # -------------------------------------------------------------------------
    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Extracting temporal features …")
        out = pd.DataFrame(index=df.index)

        ts = df["timestamp_ms"].astype(np.int64).sort_values()
        iat = ts.diff().fillna(0).reindex(df.index).astype(np.float32)
        out["inter_arrival_ms"] = iat.clip(lower=0, upper=3_600_000)

        # Rolling burstiness (per-window)
        w = min(self.window_size, max(len(df) // 10, 3))
        out["burstiness"] = (
            df["timestamp_ms"]
            .rolling(window=w, min_periods=3)
            .apply(lambda x: burstiness_score(x.values.astype(np.float64)),
                   raw=False)
            .fillna(0.0)
        )

        # Periodicity via autocorrelation of inter-arrival times
        out["periodicity_score"] = periodicity_score(
            df["timestamp_ms"].values.astype(np.float64)
        )

        # Hour-of-day entropy
        hours = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.hour
        global_hour_entropy = shannon_entropy(hours.astype(str).tolist(), base=2.0)
        out["hour_of_day_entropy"] = global_hour_entropy

        return out

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame containing all engineered feature columns."""
        parts: list[pd.DataFrame] = []
        if self.enable_dns:
            parts.append(self._dns_features(df))
        if self.enable_tls:
            parts.append(self._tls_features(df))
        if self.enable_doh:
            parts.append(self._doh_features(df))
        if self.enable_temporal:
            parts.append(self._temporal_features(df))

        features = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
        if "label" in df.columns:
            features["label"] = df["label"].astype(np.int8)
        return features


# =============================================================================
# 4. DIMENSIONALITY REDUCTION (PCA + t-SNE, Sec. 3.1)
# =============================================================================
@dataclass
class Reducer:
    pca_components: int = 64
    pca_whiten: bool = True
    tsne_components: int = 2
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    random_state: int = 42

    pca: Optional[PCA] = field(default=None, init=False)
    scaler: Optional[StandardScaler] = field(default=None, init=False)
    tsne: Optional[TSNE] = field(default=None, init=False)

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, do_tsne: bool = False
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        # PCA — "remove collinear features at the cost of not losing important
        # structural information" (Sec. 3.1)
        n_components = min(self.pca_components, Xs.shape[1], Xs.shape[0])
        self.pca = PCA(
            n_components=n_components,
            whiten=self.pca_whiten,
            random_state=self.random_state,
        )
        X_pca = self.pca.fit_transform(Xs)
        logger.info(
            "PCA: %d → %d components (explained variance = %.4f)",
            X.shape[1],
            n_components,
            float(self.pca.explained_variance_ratio_.sum()),
        )

        X_tsne: Optional[np.ndarray] = None
        if do_tsne and Xs.shape[0] > self.tsne_perplexity + 1:
            self.tsne = TSNE(
                n_components=self.tsne_components,
                perplexity=self.tsne_perplexity,
                n_iter=self.tsne_n_iter,
                init="pca",
                random_state=self.random_state,
                n_jobs=-1,
            )
            X_tsne = self.tsne.fit_transform(X_pca)
            logger.info("t-SNE: produced 2-D visualization embedding.")

        return X_pca, X_tsne

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Reducer not fitted. Call fit_transform() first.")
        return self.pca.transform(self.scaler.transform(X))


# =============================================================================
# 5. CLASS IMBALANCE — SMOTE
# =============================================================================
def apply_smote(
    X: np.ndarray, y: np.ndarray, k_neighbors: int = 5, sampling_strategy: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    if not _SMOTE_AVAILABLE:
        logger.warning("SMOTE skipped (imbalanced-learn not installed).")
        return X, y
    sm = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=42,
    )
    X_res, y_res = sm.fit_resample(X, y)
    logger.info("SMOTE: %d → %d samples", len(y), len(y_res))
    return X_res, y_res


# =============================================================================
# 6. KL-DIVERGENCE DATASET-BIAS TEST (Table 1, row 4)
# =============================================================================
def kl_divergence_pq(
    p: np.ndarray, q: np.ndarray, bins: int = 50, eps: float = 1e-9
) -> float:
    """
    D_KL(P || Q) = Σ p_i log(p_i / q_i).
    Used in Table 1 to detect distributional skew between splits.
    """
    p_hist, edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def dataset_bias_report(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    X_synth: Optional[np.ndarray] = None,
    threshold: float = 0.15,
) -> dict:
    """Per-feature KL divergence of val/test/synth vs train."""
    report: dict = {"threshold": threshold, "features": {}}
    for i in range(X_train.shape[1]):
        row = {
            "val_kl": kl_divergence_pq(X_train[:, i], X_val[:, i]),
            "test_kl": kl_divergence_pq(X_train[:, i], X_test[:, i]),
        }
        if X_synth is not None:
            row["synth_kl"] = kl_divergence_pq(X_train[:, i], X_synth[:, i])
        row["biased"] = max(row.values()) > threshold
        report["features"][f"f{i:03d}"] = row
    n_biased = sum(1 for v in report["features"].values() if v["biased"])
    report["n_biased_features"] = n_biased
    report["overall_pass"] = n_biased == 0
    logger.info(
        "Dataset-bias check: %d / %d features exceed KL threshold %.2f",
        n_biased,
        X_train.shape[1],
        threshold,
    )
    return report


# =============================================================================
# 7. ORCHESTRATOR
# =============================================================================
@dataclass
class PreprocessingPipeline:
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    reducer: Reducer = field(default_factory=Reducer)

    def run(
        self,
        raw_df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        smote: bool = True,
        kl_threshold: float = 0.15,
        do_tsne: bool = False,
    ) -> dict:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        logger.info("Step 1/5 — feature engineering (%d rows) …", len(raw_df))
        feat_df = self.extractor.transform(raw_df)

        label = feat_df.pop("label").to_numpy(dtype=np.int64)
        X_raw = feat_df[ALL_ENGINEERED_COLS].fillna(0.0).to_numpy(dtype=np.float32)

        logger.info("Step 2/5 — stratified train/val/test split …")
        rng = np.random.default_rng(42)
        idx = np.arange(len(X_raw))
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

        logger.info("Step 3/5 — PCA reduction …")
        X_tr, X_tsne = self.reducer.fit_transform(X_raw[tr], label[tr], do_tsne=do_tsne)
        X_va = self.reducer.transform(X_raw[va])
        X_te = self.reducer.transform(X_raw[te])

        logger.info("Step 4/5 — dataset-bias (KL divergence) check …")
        bias = dataset_bias_report(X_tr, X_va, X_te, threshold=kl_threshold)

        logger.info("Step 5/5 — SMOTE class balancing …")
        y_tr, y_va, y_te = label[tr], label[va], label[te]
        if smote:
            X_tr, y_tr = apply_smote(X_tr, y_tr)

        return {
            "X_train": X_tr, "y_train": y_tr,
            "X_val": X_va, "y_val": y_va,
            "X_test": X_te, "y_test": y_te,
            "X_tsne": X_tsne,
            "feature_names": ALL_ENGINEERED_COLS,
            "bias_report": bias,
            "reducer": self.reducer,
        }


# =============================================================================
# 8. CLI
# =============================================================================
def _cli() -> None:
    p = argparse.ArgumentParser(description="DoH preprocessing pipeline (Sec. 3.1)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pca-components", type=int, default=64)
    p.add_argument("--no-smote", action="store_true")
    p.add_argument("--do-tsne", action="store_true")
    p.add_argument("--kl-threshold", type=float, default=0.15)
    args = p.parse_args()

    logger.info("Loading raw data from %s", args.input)
    raw = pd.read_parquet(args.input) if args.input.suffix == ".parquet" \
        else pd.read_csv(args.input)

    pipe = PreprocessingPipeline(
        reducer=Reducer(pca_components=args.pca_components)
    )
    out = pipe.run(
        raw,
        smote=not args.no_smote,
        do_tsne=args.do_tsne,
        kl_threshold=args.kl_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {f"f{i:03d}": out["X_train"][:, i] for i in range(out["X_train"].shape[1])}
    )
    out_df["label"] = out["y_train"]
    out_df.to_parquet(args.output, index=False)
    logger.success("Saved %d rows × %d features → %s",
                   len(out_df), out["X_train"].shape[1], args.output)


if __name__ == "__main__":
    _cli()
