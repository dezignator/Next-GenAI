"""
src/models/hybrid_model.py
==========================
Hybrid ensemble: CapsNet + Graph Transformer + CSSL.

Paper reference (Abstract, Fig. 1):
    "Capsule Networks (CapsNets) are used to learn hierarchical traffic flow
     patterns, Graph Transformers to uncover temporal anomalies, and
     Contrastive Self-Supervised Learning (CSSL) to leverage massive unlabeled
     datasets."

Ensemble weights are chosen to reflect the reported accuracies in Fig. 1:
    CapsNet   : 99.1% accuracy  → weight 0.45
    GraphTF   : 98.8% recall    → weight 0.40
    CSSL      : 0.992 AUC       → weight 0.15

Fusion strategies supported:
    - weighted_average    (default; matches paper)
    - stacking            (learned meta-classifier)
    - attention_fusion    (learnable per-sample attention)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.capsule_network import CapsNet, CapsNetConfig, MarginLoss
from src.models.cssl import CSSL, CSSLConfig
from src.models.graph_transformer import (
    GraphTransformer,
    GraphTransformerConfig,
    build_knn_graph,
)


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class HybridConfig:
    # Sub-model configs
    capsnet: CapsNetConfig = field(default_factory=CapsNetConfig)
    graph_transformer: GraphTransformerConfig = field(default_factory=GraphTransformerConfig)
    cssl: CSSLConfig = field(default_factory=CSSLConfig)

    # Ensemble
    strategy: Literal["weighted_average", "stacking", "attention_fusion"] = "weighted_average"
    weights: tuple[float, float, float] = (0.45, 0.40, 0.15)   # Fig. 1 accuracies
    threshold: float = 0.5
    num_classes: int = 2

    # Fusion layer (used only for stacking/attention_fusion)
    fusion_hidden: int = 64
    fusion_dropout: float = 0.1


# =============================================================================
# 2. ATTENTION FUSION HEAD
# =============================================================================
class AttentionFusion(nn.Module):
    """
    Learns per-sample attention weights over the three model outputs.

    Given logits (B, 3, C) from CapsNet, GraphTF, CSSL-classifier,
    produce fused logits (B, C) and interpretable attention (B, 3).
    """

    def __init__(self, num_models: int = 3, num_classes: int = 2, hidden: int = 16) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(num_classes, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.num_models = num_models

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        logits : (B, M, C)
        returns : fused_logits (B, C), attention (B, M)
        """
        scores = self.score(logits).squeeze(-1)          # (B, M)
        attn = F.softmax(scores, dim=-1)                 # (B, M)
        fused = (attn.unsqueeze(-1) * logits).sum(dim=1)  # (B, C)
        return fused, attn


# =============================================================================
# 3. STACKING META-CLASSIFIER
# =============================================================================
class StackingHead(nn.Module):
    """
    Meta-classifier trained on concatenated logits of the 3 models.
    Input:  (B, M·C)
    Output: (B, C)
    """

    def __init__(self, num_models: int = 3, num_classes: int = 2,
                 hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        in_dim = num_models * num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        B = logits.shape[0]
        return self.net(logits.view(B, -1))


# =============================================================================
# 4. HYBRID ENSEMBLE
# =============================================================================
class HybridModel(nn.Module):
    """
    Combines CapsNet, Graph Transformer, and CSSL (as a frozen embedding
    + linear classifier) via one of three fusion strategies.

    Forward signature (graph input optional):
        logits, aux = model(
            x_tabular,          # (B, input_dim) for CapsNet + CSSL
            x_graph,            # (B, N, d_model) for GraphTF — optional
            A,                  # (B, N, N) or (N, N) adjacency — optional
        )
    """

    def __init__(self, cfg: HybridConfig = HybridConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        # Sub-models
        self.capsnet = CapsNet(cfg.capsnet)
        self.graph_tf = GraphTransformer(cfg.graph_transformer)
        self.cssl = CSSL(cfg.cssl)

        # CSSL → classifier on top of learned embedding
        self.cssl_classifier = nn.Linear(cfg.cssl.encoder_hidden, cfg.num_classes)

        # Fusion
        if cfg.strategy == "weighted_average":
            self.fusion = None
        elif cfg.strategy == "stacking":
            self.fusion = StackingHead(
                num_models=3,
                num_classes=cfg.num_classes,
                hidden=cfg.fusion_hidden,
                dropout=cfg.fusion_dropout,
            )
        elif cfg.strategy == "attention_fusion":
            self.fusion = AttentionFusion(
                num_models=3,
                num_classes=cfg.num_classes,
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {cfg.strategy}")

        # Losses
        self.margin_loss = MarginLoss()

    # -------------------------------------------------------------------------
    # Sub-model forward helpers
    # -------------------------------------------------------------------------
    def _run_capsnet(self, x_tab: torch.Tensor) -> torch.Tensor:
        """CapsNet outputs class probabilities (‖v_k‖)."""
        return self.capsnet(x_tab)                       # (B, C)

    def _run_cssl(self, x_tab: torch.Tensor) -> torch.Tensor:
        """CSSL embedding → linear classifier → logits."""
        emb = self.cssl.get_embedding(x_tab)             # (B, hidden)
        return self.cssl_classifier(emb)                 # (B, C)

    def _run_graph_tf(
        self,
        x_graph: Optional[torch.Tensor],
        A: Optional[torch.Tensor],
        fallback_batch: int,
    ) -> torch.Tensor:
        """
        GraphTF logits. If graph input is missing (e.g., single-flow inference),
        returns uniform logits so ensemble weights still sum to 1.
        """
        if x_graph is None or A is None:
            return torch.zeros(fallback_batch, self.cfg.num_classes,
                               device=next(self.parameters()).device)
        logits, _ = self.graph_tf(x_graph, A)
        return logits

    # -------------------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------------------
    def _fuse(
        self,
        p_caps: torch.Tensor,      # (B, C) probs
        logits_gt: torch.Tensor,   # (B, C) logits
        logits_cssl: torch.Tensor, # (B, C) logits
    ) -> tuple[torch.Tensor, dict]:
        """
        All three outputs are combined here.

        For weighted_average: convert everything to probability space,
        average with paper weights (Fig. 1).
        """
        p_gt = F.softmax(logits_gt, dim=-1)
        p_cssl = F.softmax(logits_cssl, dim=-1)
        stack = torch.stack([p_caps, p_gt, p_cssl], dim=1)    # (B, 3, C)

        if self.cfg.strategy == "weighted_average":
            w = torch.tensor(self.cfg.weights, device=p_caps.device,
                             dtype=p_caps.dtype).view(1, 3, 1)
            fused_prob = (stack * w).sum(dim=1)                # (B, C)
            return fused_prob, {"weights": self.cfg.weights}

        if self.cfg.strategy == "stacking":
            fused_logits = self.fusion(stack)                  # (B, C)
            return F.softmax(fused_logits, dim=-1), {"weights": "learned"}

        if self.cfg.strategy == "attention_fusion":
            fused_logits, attn = self.fusion(stack)
            return F.softmax(fused_logits, dim=-1), {"attention": attn.detach()}

        raise RuntimeError("Unreachable")

    # -------------------------------------------------------------------------
    # Public forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        x_tab: torch.Tensor,
        x_graph: Optional[torch.Tensor] = None,
        A: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ):
        """
        Returns
        -------
        probs : (B, C) — fused malicious/benign probabilities
        aux   : dict with keys
                  'capsnet'     : (B, C)
                  'graph_tf'    : (B, C)
                  'cssl'        : (B, C)
                  'aux_weights' : dict
        """
        p_caps = self._run_capsnet(x_tab)                       # (B, C)
        logits_gt = self._run_graph_tf(x_graph, A, x_tab.shape[0])
        logits_cssl = self._run_cssl(x_tab)

        fused_prob, aux_w = self._fuse(p_caps, logits_gt, logits_cssl)

        if not return_aux:
            return fused_prob

        aux = {
            "capsnet":  p_caps,
            "graph_tf": F.softmax(logits_gt, dim=-1),
            "cssl":     F.softmax(logits_cssl, dim=-1),
            "aux_weights": aux_w,
        }
        return fused_prob, aux

    # -------------------------------------------------------------------------
    # Convenience: pretrain CSSL on unlabeled DoH traffic
    # -------------------------------------------------------------------------
    def pretrain_cssl(
        self, unlabeled_loader, optimizer: torch.optim.Optimizer,
        epochs: int = 100,
    ) -> list[float]:
        """
        Pretrain the CSSL encoder on unlabeled DoH data (Sec. 3.5).
        Freezes CapsNet & GraphTF. Returns per-epoch loss history.
        """
        self.capsnet.eval()
        self.graph_tf.eval()
        self.cssl.train()
        history: list[float] = []

        for ep in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in unlabeled_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                loss, _ = self.cssl(x)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
                n += x.size(0)
            epoch_loss /= max(n, 1)
            history.append(epoch_loss)
            print(f"[CSSL] epoch {ep + 1}/{epochs}  loss={epoch_loss:.4f}")

        return history

    # -------------------------------------------------------------------------
    # Training step (joint loss)
    # -------------------------------------------------------------------------
    def compute_loss(
        self,
        probs: torch.Tensor,
        aux: dict,
        y: torch.Tensor,
        cssl_weight: float = 0.1,
        ce_weight: float = 0.5,
    ) -> tuple[torch.Tensor, dict]:
        """
        Joint loss:
            L_total = L_margin(capsnet) + ce_weight·L_CE(fused) + cssl_weight·L_CSSL
        """
        L_margin = self.margin_loss(aux["capsnet"], y)
        L_ce = F.cross_entropy(torch.log(probs + 1e-9), y)
        L_cssl = torch.tensor(0.0, device=probs.device)   # optional external

        total = L_margin + ce_weight * L_ce + cssl_weight * L_cssl
        return total, {"margin": L_margin.item(), "ce": L_ce.item(),
                       "cssl": L_cssl.item(), "total": total.item()}


# =============================================================================
# 5. SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = HybridConfig(
        capsnet=CapsNetConfig(input_dim=64),
        graph_transformer=GraphTransformerConfig(d_model=64, num_layers=2),
        cssl=CSSLConfig(encoder_in_dim=64),
    )
    model = HybridModel(cfg)

    B, N = 4, 12
    x_tab = torch.randn(B, 64)
    x_graph = torch.randn(B, N, 64)
    A = build_knn_graph(x_graph[0], k=4)
    y = torch.randint(0, 2, (B,))

    probs, aux = model(x_tab, x_graph, A)
    loss, comps = model.compute_loss(probs, aux, y)
    print(f"probs={probs.shape}  loss={loss.item():.4f}")
    print(f"components={comps}")
    print(f"aux_weights={aux['aux_weights']}")
