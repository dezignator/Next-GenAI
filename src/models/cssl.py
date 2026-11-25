"""
src/models/cssl.py
==================
Contrastive Self-Supervised Learning (CSSL) for zero-day generalization.

Paper reference (Sec. 3.5, Eq. 15):
    L_CSSL = - Σ_{(x,x⁺)} log[ exp(sim(f(x), f(x⁺))/τ)
                              / Σ_{x⁻} exp(sim(f(x), f(x⁻))/τ) ]

Reported AUC: 0.992 (Fig. 1). Enables zero-day generalization without labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. AUGMENTATIONS — produce positive pairs (x, x⁺)
# =============================================================================
class DoHAugmenter:
    """Produce positive views of a DoH feature vector (Sec. 3.5)."""

    def __init__(self, noise_std: float = 0.01, dropout_p: float = 0.1) -> None:
        self.noise_std = noise_std
        self.dropout_p = dropout_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Gaussian noise
        x1 = x + torch.randn_like(x) * self.noise_std
        # 2) Feature dropout (per-sample Bernoulli mask)
        mask = (torch.rand_like(x) > self.dropout_p).float()
        x2 = x1 * mask
        # 3) Random time-shift (roll along feature axis)
        shift = int(torch.randint(-3, 4, (1,)).item())
        return torch.roll(x2, shifts=shift, dims=-1)


# =============================================================================
# 2. PROJECTION HEAD (SimCLR-style)
# =============================================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# =============================================================================
# 3. NT-XENT LOSS  — Eq. (15)
# =============================================================================
def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    L_CSSL = - Σ_{(x,x⁺)} log[ exp(sim(z_i, z_i⁺)/τ)
                              / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]     [Eq. 15]

    z1, z2 : (B, D) — L2-normalized embeddings of the two views.
    Positive pair for i in z1 is z2[i]; all other 2B-2 rows are negatives.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                         # (2B, D)
    sim = z @ z.t() / temperature                          # (2B, 2B)

    # Mask self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Positive indices: for i in [0,B) → i+B; for i in [B,2B) → i-B
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    pos = sim[torch.arange(2 * B), pos_idx]               # (2B,)

    # Numerator/Denominator in log-space
    log_prob = pos - torch.logsumexp(sim, dim=-1)
    return -log_prob.mean()


# =============================================================================
# 4. MEMORY BANK (MoCo-style fallback)
# =============================================================================
class MemoryBank:
    def __init__(self, size: int = 4096, dim: int = 128) -> None:
        self.bank = F.normalize(torch.randn(size, dim), dim=-1)
        self.ptr = 0
        self.size = size

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        b = z.shape[0]
        end = (self.ptr + b) % self.size
        if end < self.ptr:                                # wrap
            self.bank[self.ptr:] = z[:self.size - self.ptr]
            self.bank[:end] = z[self.size - self.ptr:]
        else:
            self.bank[self.ptr:end] = z
        self.ptr = end

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.size, (n,))
        return self.bank[idx]


# =============================================================================
# 5. CSSL MODEL
# =============================================================================
@dataclass
class CSSLConfig:
    encoder_in_dim: int = 64         # matches PCA output dim
    encoder_hidden: int = 128
    projection_dim: int = 128
    projection_hidden: int = 256
    temperature: float = 0.07        # Eq. (15) τ
    noise_std: float = 0.01
    dropout_p: float = 0.1


class CSSLEncoder(nn.Module):
    def __init__(self, cfg: CSSLConfig) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(cfg.encoder_in_dim, cfg.encoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.encoder_hidden, cfg.encoder_hidden),
        )
        self.head = ProjectionHead(
            cfg.encoder_hidden, cfg.projection_hidden, cfg.projection_dim
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        z = self.head(h)
        return h, z


class CSSL(nn.Module):
    """
    Full contrastive pretraining module.

    Usage:
        cssl = CSSL()
        for x in unlabeled_loader:
            loss = cssl(x)
            loss.backward()
    """

    def __init__(self, cfg: CSSLConfig = CSSLConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = CSSLEncoder(cfg)
        self.augmenter = DoHAugmenter(cfg.noise_std, cfg.dropout_p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.augmenter(x)
        x2 = self.augmenter(x)
        _, z1 = self.encoder(x1)
        _, z2 = self.encoder(x2)
        loss = nt_xent_loss(z1, z2, temperature=self.cfg.temperature)
        return loss, z1

    # Convenience accessor for downstream classifiers
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return h


# =============================================================================
# 6. SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    cssl = CSSL()
    x = torch.randn(32, 64)
    loss, z = cssl(x)
    print(f"loss={loss.item():.4f}  z={z.shape}")
