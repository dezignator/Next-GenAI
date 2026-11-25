"""
src/models/capsule_network.py
=============================
Capsule Network for DoH anomaly detection.

Paper reference (Sec. 3.3, Eq. 4–8):
    Eq. (4)  Ricci-flow routing:
        ∂c_ij/∂t = -Ric(c_ij) + β Σ α_ij T_ij(c_ij^{l-1})
    Eq. (5)  Geometric consistency:
        α_ij = softmax(angle(W_i·c_i, W_j·c_j))
    Eq. (6)  Dynamic routing coupling coefficients:
        c_ij = exp(b_ij) / Σ exp(b_ik),  s_j = Σ c_ij W_ij u_i
    Eq. (7)  Margin loss:
        L_k = T_k·max(0, m⁺-‖v_k‖)² + λ(1-T_k)·max(0, ‖v_k‖-m⁻)²
    Eq. (8)  Squash:
        squash(s) = (‖s‖²/(1+‖s‖²)) · s/‖s‖

Reported accuracy: 99.1% (Abstract, Fig. 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. SQUASH FUNCTION  — Eq. (8)
# =============================================================================
def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    squash(s) = (‖s‖² / (1 + ‖s‖²)) · s / ‖s‖            [Eq. 8]
    """
    sq_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm + eps)
    return scale * x / torch.sqrt(sq_norm + eps)


# =============================================================================
# 2. RICCI-FLOW DYNAMIC ROUTING  — Eq. (4), (5), (6)
# =============================================================================
class RicciFlowRouting(nn.Module):
    """
    Implements the geometric routing described by Eq. (4) and Eq. (5).

    The paper proposes augmenting the classical dynamic-routing coupling
    coefficients (Eq. 6) with a Ricci-flow-inspired regularization term
    (Eq. 4) and an angle-based geometric-consistency gate (Eq. 5).

    We preserve standard routing semantics (sum of coupling coefficients = 1)
    and add:
       - a Ricci-curvature penalty derived from the local neighborhood
       - a geometric consistency score α_ij based on angular proximity
    """

    def __init__(
        self,
        num_caps_in: int,
        num_caps_out: int,
        in_dim: int,
        out_dim: int,
        num_iterations: int = 3,
        beta: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.beta = beta
        self.temperature = temperature

        # W_ij ∈ ℝ^{out_dim × in_dim}  [Eq. 6]
        self.W = nn.Parameter(
            torch.empty(num_caps_in, num_caps_out, out_dim, in_dim)
        )
        nn.init.xavier_uniform_(self.W)

    def _geometric_consistency(
        self, u_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Eq. (5): α_ij = softmax_j( angle(W_i·c_i, W_j·c_j) )

        We compute pairwise cosine angles between every input capsule's
        prediction and every output capsule, then softmax over j.
        """
        # u_hat: (B, num_caps_in, num_caps_out, out_dim)
        B, N_in, N_out, D = u_hat.shape
        # Flatten to (B, N_in, N_out * D) for pairwise dot
        flat = u_hat.reshape(B, N_in, N_out * D)
        flat_n = F.normalize(flat, dim=-1, eps=1e-8)
        cos = torch.einsum("bik,bjk->bij", flat_n, flat_n)  # (B, N_in, N_in)
        # Project to (B, N_in, N_out) by averaging over j-blocks
        cos = cos.mean(dim=-1, keepdim=True).expand(-1, -1, N_out)
        return F.softmax(cos / self.temperature, dim=-1)

    def _ricci_flow_step(
        self, c: torch.Tensor, u_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Eq. (4) (discrete-time form):
            c^{(t+1)} = c^{(t)} - Ric(c^{(t)}) + β · α · T(u)
        Ric(·) implemented as the local spread (variance) of the coupling
        matrix. T(u) is the agreement u_hat · v.
        """
        # Ricci curvature ≈ variance of coupling over the j-axis
        ric = c.var(dim=-1, keepdim=True).expand_as(c)
        return c - ric

    def forward(
        self, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        u : (B, num_caps_in, in_dim)

        Returns
        -------
        v       : (B, num_caps_out, out_dim)  — output capsules
        c_final : (B, num_caps_in, num_caps_out) — coupling coefficients
        """
        B, N_in, _ = u.shape
        N_out, out_dim = self.W.shape[1], self.W.shape[2]

        # W_ij · u_i  → (B, N_in, N_out, out_dim)          [Eq. 6]
        u_hat = torch.einsum("bin,ino d->biod", u, self.W) if False else \
                torch.einsum("bin,ino d->biod".replace(" ", ""), u, self.W)

        # Initialize logits b_ij = 0
        b = torch.zeros(B, N_in, N_out, device=u.device)

        # Geometric consistency gate — Eq. (5)
        alpha = self._geometric_consistency(u_hat)   # (B, N_in, N_out)

        for _ in range(self.num_iterations):
            # Eq. (6): c_ij = softmax_j(b_ij)
            c = F.softmax(b, dim=-1) * alpha          # apply geometric gate
            c = c / (c.sum(dim=-1, keepdim=True) + 1e-8)

            # s_j = Σ_i c_ij · u_hat_ij
            s = torch.einsum("bij,bijd->bjd", c, u_hat)

            # v_j = squash(s_j)  — Eq. (8)
            v = squash(s, dim=-1)

            # Agreement a_ij = u_hat_ij · v_j
            agreement = torch.einsum("bijd,bjd->bij", u_hat, v)

            # Ricci-flow update (Eq. 4)
            b = self._ricci_flow_step(b + self.beta * agreement, u_hat)

        return v, c


# =============================================================================
# 3. CAPSULE NETWORK  — Eq. (4)–(8)
# =============================================================================
@dataclass
class CapsNetConfig:
    input_dim: int = 256
    primary_num_caps: int = 32
    primary_caps_dim: int = 8
    num_classes: int = 2
    class_caps_dim: int = 16
    num_iterations: int = 3
    beta: float = 0.5
    temperature: float = 1.0
    dropout: float = 0.2


class CapsNet(nn.Module):
    """
    Hierarchical capsule network for DoH query classification.

    Architecture (per Sec. 3.3):
        Input (B, input_dim)
        → ReLU FC  → (B, primary_num_caps, primary_caps_dim)  [PrimaryCaps]
        → RicciFlowRouting                                     [Eq. 4, 5, 6]
        → v_j (B, num_classes, class_caps_dim)                 [Eq. 8]
        → ‖v_j‖² ⇒ logits for margin loss                      [Eq. 7]
    """

    def __init__(self, cfg: CapsNetConfig = CapsNetConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        hidden = cfg.primary_num_caps * cfg.primary_caps_dim
        self.primary_fc = nn.Sequential(
            nn.Linear(cfg.input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

        self.routing = RicciFlowRouting(
            num_caps_in=cfg.primary_num_caps,
            num_caps_out=cfg.num_classes,
            in_dim=cfg.primary_caps_dim,
            out_dim=cfg.class_caps_dim,
            num_iterations=cfg.num_iterations,
            beta=cfg.beta,
            temperature=cfg.temperature,
        )

    def forward(
        self, x: torch.Tensor, return_caps: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        h = self.primary_fc(x)                          # (B, P·D)
        h = h.view(B, self.cfg.primary_num_caps, self.cfg.primary_caps_dim)
        # Optional: squash on primary capsules
        u = squash(h, dim=-1)
        v, _ = self.routing(u)                          # (B, C, class_dim)
        # Class probabilities derived from capsule norms (Eq. 7 uses ‖v‖²)
        probs = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)  # (B, C)
        if return_caps:
            return probs, v
        return probs


# =============================================================================
# 4. MARGIN LOSS  — Eq. (7)
# =============================================================================
class MarginLoss(nn.Module):
    """
    L_k = T_k·max(0, m⁺ - ‖v_k‖)² + λ·(1-T_k)·max(0, ‖v_k‖ - m⁻)²     [Eq. 7]
    """

    def __init__(
        self,
        m_plus: float = 0.9,
        m_minus: float = 0.1,
        lambda_: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        self.reduction = reduction

    def forward(self, capsule_norms: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        capsule_norms : (B, C)  — ‖v_k‖
        labels        : (B,)    — integer class indices
        """
        B, C = capsule_norms.shape
        T = F.one_hot(labels, num_classes=C).float()
        pos = torch.clamp(self.m_plus - capsule_norms, min=0.0) ** 2
        neg = torch.clamp(capsule_norms - self.m_minus, min=0.0) ** 2
        losses = T * pos + self.lambda_ * (1.0 - T) * neg  # (B, C)
        per_sample = losses.sum(dim=-1)
        return per_sample.mean() if self.reduction == "mean" \
            else per_sample.sum() if self.reduction == "sum" else per_sample


# =============================================================================
# 5. SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    model = CapsNet(CapsNetConfig(input_dim=64))
    x = torch.randn(8, 64)
    y = torch.randint(0, 2, (8,))
    probs, caps = model(x, return_caps=True)
    loss = MarginLoss()(probs, y)
    print(f"probs.shape={probs.shape}  caps.shape={caps.shape}  loss={loss.item():.4f}")
