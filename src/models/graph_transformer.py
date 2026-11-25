"""
src/models/graph_transformer.py
===============================
Graph Transformer for temporal anomaly detection in DoH traffic.

Paper reference (Sec. 3.4, Eq. 9–14):
    Eq. (9)   α_ij = softmax( (Qh_i)ᵀ(Kh_j) / √d_k )
    Eq. (10)  S = ‖H_normal - H_test‖_F
    Eq. (11)  α_ij = exp(h_iᵀ W_a h_j) / Σ_k exp(h_iᵀ W_a h_k)
    Eq. (12)  h'_i = σ( Σ_{j∈N(i)} α_ij W_a h_j )
    Eq. (13)  sheaf morphism Φ: ⊕F(v) → ⊕F(e)
    Eq. (14)  ∇_A = A W_Q K W_Kᵀ / √d + λ L_F

Reported recall: 98.8% (Abstract, Fig. 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. GRAPH CONSTRUCTION
# =============================================================================
def build_knn_graph(
    x: torch.Tensor, k: int = 10, threshold: float = 0.0
) -> torch.Tensor:
    """
    Build symmetric k-NN adjacency from node features.

    Parameters
    ----------
    x : (N, d)
    k : neighbors per node
    threshold : cosine similarity floor (0 → keep all k)

    Returns
    -------
    A : (N, N) — edge weights in [0, 1]
    """
    n = x.shape[0]
    xn = F.normalize(x, dim=-1, eps=1e-8)
    sim = xn @ xn.t()                                        # (N, N)
    sim.fill_diagonal_(-float("inf"))
    topk = sim.topk(k=k, dim=-1).indices                    # (N, k)
    A = torch.zeros_like(sim)
    A.scatter_(1, topk, 1.0)
    A = ((A + A.t()) > 0).float() * (sim * 0.5 + sim.t() * 0.5).clamp(min=0)
    if threshold > 0:
        A = A * (A >= threshold).float()
    return A


# =============================================================================
# 2. MULTI-HEAD GRAPH ATTENTION  — Eq. (9), (11), (12)
# =============================================================================
class GraphAttention(nn.Module):
    """
    Multi-head attention with graph masking.

    Eq. (9) is standard scaled dot-product attention; Eq. (11) is its
    add-a-bias (W_a) graph-attention form; Eq. (12) is the σ(Σ α_ij W_a h_j)
    aggregation.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_a = nn.Linear(d_model, d_model, bias=False)   # Eq. (11) — W_a
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()                          # Eq. (12) σ ∈ {ReLU, GELU}

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        h : (B, N, d_model)   — hidden states
        A : (B, N, N) or (N, N) — adjacency mask (additive bias)
        """
        B, N, D = h.shape
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(B, -1, -1)
        q = self.W_q(h).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(h).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(h).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # Eq. (9): scaled dot-product scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)    # (B, H, N, N)

        # Eq. (11): additive W_a bias masked by A
        wa_bias = self.W_a(h)                                        # (B, N, D)
        wa_bias = wa_bias.mean(dim=-1, keepdim=True)                 # (B, N, 1)
        wa_bias = wa_bias.expand(-1, -1, N).unsqueeze(1)             # (B, 1, N, N)
        mask = (A > 0).unsqueeze(1)                                  # (B, 1, N, N)
        scores = scores + wa_bias
        scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Eq. (12): σ(Σ α_ij W_a h_j)
        out = attn @ v                                               # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.activation(self.W_o(out))
        return out


# =============================================================================
# 3. SHEAF LAPLACIAN  — Eq. (13), (14)
# =============================================================================
class SheafLaplacian(nn.Module):
    """
    Simplified cellular-sheaf Laplacian L_F used in Eq. (14).

    Full sheaf theory assigns to each vertex v a stalk F(v) ∈ ℝ^d and to each
    edge e a restriction map ρ_{v⊲e} : F(v) → F(e). The sheaf Laplacian is
    L = δᵀ δ, where δ is the sheaf coboundary operator.

    We implement a learnable restriction map per-edge in the same stalk dim
    as the model. This preserves the sheaf-Laplacian structure while remaining
    tractable at scale (max_nodes_per_graph = 5000).
    """

    def __init__(self, stalk_dim: int, num_edges: int, reg: float = 0.1) -> None:
        super().__init__()
        self.stalk_dim = stalk_dim
        self.reg = reg
        # Learnable per-edge restriction maps ρ_e : F(v) → F(e)
        self.rho = nn.Parameter(torch.empty(num_edges, stalk_dim, stalk_dim))
        nn.init.orthogonal_(self.rho)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        h          : (N, D)
        edge_index : (2, E) — (src, dst)
        returns LF : (N, D) — sheaf-Laplacian regularizer applied to h
        """
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]
        D = h.shape[-1]
        # ρ_e · h_v  for each edge
        # self.rho is (num_edges_max, D, D); slice to actual E
        rho = self.rho[:E]                                            # (E, D, D)
        h_src = h[src]                                                # (E, D)
        h_dst = h[dst]
        h_src_r = torch.bmm(rho, h_src.unsqueeze(-1)).squeeze(-1)     # (E, D)
        h_dst_r = torch.bmm(rho, h_dst.unsqueeze(-1)).squeeze(-1)

        # Coboundary δ(h)_e = ρ_{src,e} h_src - ρ_{dst,e} h_dst
        diff = h_src_r - h_dst_r                                       # (E, D)

        # LF[v] = Σ_{e incident to v} ρᵀ diff_e
        LF = torch.zeros_like(h)
        LF.index_add_(0, src, torch.bmm(rho.transpose(1, 2), diff.unsqueeze(-1)).squeeze(-1))
        LF.index_add_(0, dst, -torch.bmm(rho.transpose(1, 2), diff.unsqueeze(-1)).squeeze(-1))
        return self.reg * LF


# =============================================================================
# 4. GRAPH TRANSFORMER LAYER
# =============================================================================
class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        sheaf_reg: float = 0.1,
        sheaf_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.sheaf_enabled = sheaf_enabled
        self.attn = GraphAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        if sheaf_enabled:
            self.sheaf = SheafLaplacian(stalk_dim=d_model, num_edges=50_000, reg=sheaf_reg)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Eq. (9)–(12) + residual
        h = self.norm1(h + self.dropout(self.attn(h, A)))
        # Eq. (14): + λ L_F (sheaf Laplacian)
        if self.sheaf_enabled:
            edge_index = A[0].nonzero(as_tuple=False).t()
            h = h + self.sheaf(h[0], edge_index).unsqueeze(0)
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h


# =============================================================================
# 5. GRAPH TRANSFORMER MODEL  — Eq. (9)–(14)
# =============================================================================
@dataclass
class GraphTransformerConfig:
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_classes: int = 2
    sheaf_enabled: bool = True
    sheaf_reg: float = 0.1
    k_neighbors: int = 10


class GraphTransformer(nn.Module):
    def __init__(self, cfg: GraphTransformerConfig = GraphTransformerConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                cfg.d_model, cfg.num_heads, cfg.dim_feedforward,
                cfg.dropout, cfg.sheaf_reg, cfg.sheaf_enabled,
            ) for _ in range(cfg.num_layers)
        ])
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_classes),
        )

    def forward(
        self, x: torch.Tensor, A: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, N, d_model)
        A : (B, N, N) or (N, N)
        """
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(x.shape[0], -1, -1)
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h, A)
        # Pool over nodes
        pooled = h.mean(dim=1)                          # (B, d_model)
        logits = self.cls_head(pooled)
        return logits, h


# =============================================================================
# 6. ANOMALY SCORE  — Eq. (10)
# =============================================================================
def frobenius_anomaly_score(
    H_normal: torch.Tensor, H_test: torch.Tensor
) -> torch.Tensor:
    """
    S = ‖H_normal - H_test‖_F     [Eq. 10]
    Both inputs: (B, N, D) — per-batch Frobenius norm.
    """
    diff = H_normal - H_test
    return torch.linalg.matrix_norm(diff, ord="fro")


# =============================================================================
# 7. SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 12, 128
    x = torch.randn(2, N, D)
    A = build_knn_graph(x[0], k=4)
    model = GraphTransformer(GraphTransformerConfig(d_model=D, num_layers=2))
    logits, h = model(x, A)
    print(f"logits={logits.shape}  h={h.shape}")
    score = frobenius_anomaly_score(h, h + 0.01)
    print(f"anomaly_score={score}")
