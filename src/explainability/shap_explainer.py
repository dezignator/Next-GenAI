"""
src/explainability/shap_explainer.py
====================================
SHAP-based explainability for DoH anomaly detection.

Paper reference (Sec. 3.6, Eq. 16):
    φ_i = Σ_{S ⊆ F \\ {i}} |S|!(|F| - |S| - 1)! / |F|!
          · [ f(S ∪ {i}) - f(S) ]

Also Sec. 2 (abstract):
    "Interpretability is enhanced via SHAP and LIME, rendering AI decision-
     making processes more intelligible to analysts."
    "SHAP + LIME Interpretation → Actionable Threat Intelligence → 0.001% FPR"

Implements Eq. (16) exactly, with KernelSHAP as the sampling approximator
(the only computationally tractable estimator for deep ensembles).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed → falling back to exact Shapley enumeration.")


# =============================================================================
# 1. EQUATION 16 — EXACT SHAPLEY VALUE (small feature sets only)
# =============================================================================
def shapley_value_exact(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    baseline: np.ndarray,
    feature_index: int,
) -> float:
    """
    Exact Shapley value of feature `feature_index` for prediction f(x).

    φ_i = Σ_{S ⊆ F \\ {i}} |S|!(|F| - |S| - 1)! / |F|!
          · [ f(S ∪ {i}) - f(S) ]                            [Eq. 16]

    Parameters
    ----------
    f        : callable (D,) → scalar
    x        : (D,) point being explained
    baseline : (D,) reference "neutral" input
    feature_index : int
    """
    D = len(x)
    others = [j for j in range(D) if j != feature_index]
    phi = 0.0
    for k in range(len(others) + 1):
        weight = math.factorial(k) * math.factorial(D - k - 1) / math.factorial(D)
        for S in itertools.combinations(others, k):
            x_with = x.copy()
            x_without = x.copy()
            for j in range(D):
                if j not in S and j != feature_index:
                    x_with[j] = baseline[j]
                    x_without[j] = baseline[j]
            phi += weight * (f(x_with) - f(x_without))
    return float(phi)


# =============================================================================
# 2. HIGH-LEVEL EXPLAINER
# =============================================================================
@dataclass
class SHAPConfig:
    explainer_type: str = "kernel"          # kernel | deep | tree | gradient
    background_samples: int = 1000
    num_features_to_show: int = 10
    shapley_sampling: int = 200             # KernelSHAP num_samples
    seed: int = 42


class SHAPExplainer:
    """
    Wrapper around SHAP with:
      - torch model → predict_proba callable
      - KernelSHAP for tree-agnostic explainability
      - Fallback to exact Shapley (Eq. 16) for small feature counts (< 10)
      - Top-k feature attribution for analyst-facing reports
    """

    def __init__(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        feature_names: Optional[list[str]] = None,
        cfg: SHAPConfig = SHAPConfig(),
        device: str = "cpu",
    ) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.cfg = cfg
        self.feature_names = feature_names or [f"f{i}" for i in range(background_data.shape[1])]

        # Sub-sample background
        if len(background_data) > cfg.background_samples:
            rng = np.random.default_rng(cfg.seed)
            idx = rng.choice(len(background_data), cfg.background_samples, replace=False)
            background_data = background_data[idx]
        self.background = background_data.astype(np.float32)

        self._explainer = None
        if _SHAP_AVAILABLE:
            self._build_shap_explainer()

    # -------------------------------------------------------------------------
    # Predict function (SHAP calls this)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return malicious-class probability for each row."""
        xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        out = self.model(xt, return_aux=False) if hasattr(self.model, "forward") \
              else (self.model(xt), None)
        probs = out[0] if isinstance(out, tuple) else out
        if probs.dim() == 1:
            return probs.cpu().numpy()
        return probs[:, 1].cpu().numpy()             # P(malicious)

    # -------------------------------------------------------------------------
    # Build specific SHAP backend
    # -------------------------------------------------------------------------
    def _build_shap_explainer(self) -> None:
        try:
            if self.cfg.explainer_type == "deep":
                self._explainer = shap.DeepExplainer(self.model, self.background)
            elif self.cfg.explainer_type == "tree":
                self._explainer = shap.TreeExplainer(self.model)
            elif self.cfg.explainer_type == "gradient":
                self._explainer = shap.GradientExplainer(self.model, self.background)
            else:  # kernel (default)
                self._explainer = shap.KernelExplainer(
                    self._predict_proba, self.background
                )
            logger.info(f"SHAP explainer: {self.cfg.explainer_type}")
        except Exception as e:                                # noqa: BLE001
            logger.warning(f"SHAP init failed ({e}) → using exact Shapley fallback.")
            self._explainer = None

    # -------------------------------------------------------------------------
    # Explain one sample
    # -------------------------------------------------------------------------
    def explain(self, x: np.ndarray) -> dict:
        """
        Compute feature attributions for a single sample.

        Returns
        -------
        dict with:
            'shap_values'     : (D,) — φ_i per feature
            'base_value'      : E[f(x)] on background
            'prediction'      : f(x)
            'top_k'           : list of (feature_name, φ_i, value_i)
            'equation'        : "Eq. (16)"
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.ndim != 1:
            raise ValueError(f"Expected 1-D sample, got shape {x.shape}")

        if self._explainer is not None and _SHAP_AVAILABLE:
            return self._explain_via_shap(x)
        return self._explain_exact(x)

    # -------------------------------------------------------------------------
    # SHAP backend
    # -------------------------------------------------------------------------
    def _explain_via_shap(self, x: np.ndarray) -> dict:
        x2d = x.reshape(1, -1)
        try:
            if self.cfg.explainer_type == "kernel":
                sv = self._explainer.shap_values(
                    x2d, nsamples=self.cfg.shapley_sampling
                )
            else:
                sv = self._explainer.shap_values(x2d)

            # Normalize: some explainers return list[class] → take class 1
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            sv = np.asarray(sv).reshape(-1)

            base = self._safe_base_value()
            pred = float(self._predict_proba(x2d)[0])
            return self._format_output(x, sv, base, pred)

        except Exception as e:                                # noqa: BLE001
            logger.warning(f"SHAP failed on sample ({e}); using exact fallback.")
            return self._explain_exact(x)

    def _safe_base_value(self) -> float:
        try:
            bv = self._explainer.expected_value
            if isinstance(bv, (list, np.ndarray)):
                bv = bv[1] if len(bv) > 1 else bv[0]
            return float(bv)
        except Exception:                                     # noqa: BLE001
            return float(self._predict_proba(self.background).mean())

    # -------------------------------------------------------------------------
    # Exact Eq. (16) fallback
    # -------------------------------------------------------------------------
    def _explain_exact(self, x: np.ndarray) -> dict:
        D = len(x)
        if D > 12:
            logger.warning(
                f"Exact Shapley enumeration over {D} features is 2^{D} evaluations; "
                f"truncating to first 12 features."
            )
            D = 12
            x = x[:D]

        baseline = self.background.mean(axis=0)[:D]
        f = lambda v: float(self._predict_proba(v.reshape(1, -1))[0])  # noqa: E731
        phi = np.array(
            [shapley_value_exact(f, x, baseline, i) for i in range(D)]
        )
        base = float(self._predict_proba(self.background)[0] * 0 + np.mean(
            self._predict_proba(self.background)
        ))
        pred = f(x)
        return self._format_output(x, phi, base, pred)

    # -------------------------------------------------------------------------
    # Output formatting
    # -------------------------------------------------------------------------
    def _format_output(
        self, x: np.ndarray, shap_values: np.ndarray, base: float, pred: float
    ) -> dict:
        order = np.argsort(-np.abs(shap_values))[: self.cfg.num_features_to_show]
        top_k = [
            {
                "feature": self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                "value": float(x[i]),
                "shap_value": float(shap_values[i]),
            }
            for i in order
        ]
        return {
            "shap_values": shap_values.tolist(),
            "base_value": base,
            "prediction": pred,
            "top_k": top_k,
            "equation": "Eq. (16)",
        }

    # -------------------------------------------------------------------------
    # Batch explain (for analyst dashboards)
    # -------------------------------------------------------------------------
    def explain_batch(self, X: np.ndarray) -> list[dict]:
        return [self.explain(x) for x in X]

    # -------------------------------------------------------------------------
    # Top-k feature names (convenience)
    # -------------------------------------------------------------------------
    def top_k_features(self, explanation: dict, k: int = 3) -> list[str]:
        return [item["feature"] for item in explanation["top_k"][:k]]


# =============================================================================
# 3. LIME FALLBACK (Sec. 3.6 — LIME also required by paper)
# =============================================================================
class LIMEExplainer:
    """
    Thin wrapper around LIME for tabular explanations.
    Complements SHAP (paper requires both, Sec. 3.6).
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str],
        num_samples: int = 5000,
        num_features: int = 10,
    ) -> None:
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as e:
            raise RuntimeError("lime is required: pip install lime==0.2.0.1") from e
        self.explainer = LimeTabularExplainer(
            training_data=np.zeros((1, len(feature_names)), dtype=np.float32),
            feature_names=feature_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        self.predict_fn = predict_fn
        self.num_samples = num_samples
        self.num_features = num_features

    def explain(self, x: np.ndarray) -> dict:
        exp = self.explainer.explain_instance(
            data_row=x.reshape(-1),
            predict_fn=lambda v: np.stack(
                [1 - self.predict_fn(v), self.predict_fn(v)], axis=-1
            ),
            num_samples=self.num_samples,
            num_features=self.num_features,
        )
        return {
            "as_list": exp.as_list(),
            "intercept": exp.intercept[1],
            "score": exp.score,
            "equation_reference": "LIME (Sec. 3.6)",
        }


# =============================================================================
# 4. SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, x, return_aux=False):
            return torch.softmax(self.net(x), dim=-1)

    torch.manual_seed(0)
    model = Toy()
    bg = np.random.randn(200, 6).astype(np.float32)
    x = np.random.randn(6).astype(np.float32)
    names = ["query_entropy", "tls_mismatch", "content_type_mismatch",
             "inter_arrival_ms", "sni_mismatch", "burstiness"]

    exp = SHAPExplainer(model, bg, feature_names=names)
    result = exp.explain(x)
    print(f"Prediction: {result['prediction']:.4f}")
    print(f"Base       : {result['base_value']:.4f}")
    print(f"Top-3      : {exp.top_k_features(result, k=3)}")

    # Verify Eq. 16 additivity:  Σφ_i + base ≈ f(x)
    additivity_gap = abs(sum(result["shap_values"]) + result["base_value"] - result["prediction"])
    print(f"Additivity gap (should be ≈ 0): {additivity_gap:.6f}")
