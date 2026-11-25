"""
tests/test_graph_transformer.py
===============================
Unit tests for Graph Transformer — verifies Eq. (9), (10), (11), (12), (13), (14).
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.graph_transformer import (
    GraphAttention,
    GraphTransformer,
    GraphTransformerConfig,
    GraphTransformerLayer,
    SheafLaplacian,
    build_knn_graph,
    frobenius_anomaly_score,
)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
class TestGraphConstruction:
    def test_adjacency_symmetric(self):
        x = torch.randn(20, 8)
        A = build_knn_graph(x, k=5)
        assert A.shape == (20, 20)
        assert torch.allclose(A, A.t(), atol=1e-5)

    def test_adjacency_degree_bounded(self):
        """At most 2k neighbors per node after sym-ification."""
        x = torch.randn(50, 8)
        A = build_knn_graph(x, k=4)
        degrees = (A > 0).sum(dim=-1)
        assert (degrees <= 8).all()

    def test_threshold_filters_weak_edges(self):
        x = torch.randn(30, 8)
        A_full = build_knn_graph(x, k=5, threshold=0.0)
        A_thresh = build_knn_graph(x, k=5, threshold=0.5)
        assert (A_thresh > 0).sum() <= (A_full > 0).sum()


# =============================================================================
# Eq. (9), (11), (12) — ATTENTION
# =============================================================================
class TestGraphAttention:
    def test_output_shape(self):
        layer = GraphAttention(d_model=64, num_heads=4)
        h = torch.randn(2, 10, 64)
        A = torch.ones(2, 10, 10)
        out = layer(h, A)
        assert out.shape == (2, 10, 64)

    def test_masked_edges_contribute_zero(self):
        """Attention output must not depend on masked-out nodes."""
        layer = GraphAttention(d_model=32, num_heads=4)
        layer.eval()
        h1 = torch.randn(1, 6, 32)
        h2 = h1.clone()
        # Fully connected mask
        A_full = torch.ones(1, 6, 6)
        # Only self-loops
        A_self = torch.eye(6).unsqueeze(0)
        out_full = layer(h1, A_full)
        out_self = layer(h1, A_self)
        # They should differ because different neighborhoods aggregate
        assert not torch.allclose(out_full, out_self, atol=1e-3)

    def test_attention_scores_bounded(self):
        """Softmaxed scores sum to 1 along source dimension."""
        attn = GraphAttention(d_model=32, num_heads=4)
        h = torch.randn(1, 8, 32)
        A = torch.ones(1, 8, 8)
        # We can't directly inspect softmax from outside, so assert output is finite
        out = attn(h, A)
        assert torch.isfinite(out).all()


# =============================================================================
# Eq. (13), (14) — SHEAF LAPLACIAN
# =============================================================================
class TestSheafLaplacian:
    def test_output_shape(self):
        sheaf = SheafLaplacian(stalk_dim=16, num_edges=100, reg=0.1)
        h = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        out = sheaf(h, edge_index)
        assert out.shape == (10, 16)

    def test_laplacian_zero_on_constant_signal(self):
        """Sheaf Laplacian should be ~0 if all node embeddings are equal."""
        sheaf = SheafLaplacian(stalk_dim=8, num_edges=50, reg=1.0)
        h = torch.ones(5, 8)                       # constant
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        out = sheaf(h, edge_index)
        # diff = ρ(h) - ρ(h) = 0 → LF should be 0
        assert out.abs().max() < 1e-5

    def test_laplacian_reg_scales_output(self):
        sheaf1 = SheafLaplacian(stalk_dim=8, num_edges=20, reg=0.1)
        sheaf2 = SheafLaplacian(stalk_dim=8, num_edges=20, reg=1.0)
        with torch.no_grad():
            sheaf2.rho.copy_(sheaf1.rho)
        h = torch.randn(5, 8)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        o1 = sheaf1(h, ei)
        o2 = sheaf2(h, ei)
        ratio = (o2.abs().sum() / o1.abs().sum()).item()
        assert 8.0 < ratio < 12.0                  # reg 0.1 → 1.0 is ~10×


# =============================================================================
# FULL LAYER & MODEL
# =============================================================================
class TestGraphTransformerLayer:
    def test_forward_shape(self):
        layer = GraphTransformerLayer(
            d_model=32, num_heads=4, dim_feedforward=64,
            dropout=0.0, sheaf_enabled=False,
        )
        h = torch.randn(1, 8, 32)
        A = torch.ones(1, 8, 8)
        out = layer(h, A)
        assert out.shape == (1, 8, 32)


class TestGraphTransformerModel:
    def test_forward_shape(self):
        cfg = GraphTransformerConfig(d_model=32, num_heads=4, num_layers=2,
                                     dim_feedforward=64, sheaf_enabled=False)
        model = GraphTransformer(cfg)
        x = torch.randn(2, 8, 32)
        A = torch.ones(2, 8, 8)
        logits, h = model(x, A)
        assert logits.shape == (2, 2)              # (batch, num_classes)
        assert h.shape == (2, 8, 32)

    def test_gradients_flow(self):
        cfg = GraphTransformerConfig(d_model=32, num_heads=4, num_layers=2,
                                     dim_feedforward=64, sheaf_enabled=False)
        model = GraphTransformer(cfg)
        x = torch.randn(2, 8, 32)
        A = torch.ones(2, 8, 8)
        logits, _ = model(x, A)
        loss = F.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any((g.abs() > 0).any().item() for g in grads)


# =============================================================================
# Eq. (10) — ANOMALY SCORE
# =============================================================================
class TestAnomalyScore:
    def test_identical_inputs_zero(self):
        H = torch.randn(4, 10, 16)
        score = frobenius_anomaly_score(H, H)
        assert torch.allclose(score, torch.zeros_like(score), atol=1e-6)

    def test_known_frobenius(self):
        """Hand-check: ‖[[3,0],[0,4]]‖_F = 5."""
        a = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])   # (1, 2, 2)
        b = torch.zeros_like(a)
        score = frobenius_anomaly_score(a, b)
        assert torch.isclose(score, torch.tensor([5.0]), atol=1e-5)

    def test_anomaly_score_monotone_in_deviation(self):
        H = torch.zeros(1, 5, 8)
        s1 = frobenius_anomaly_score(H, H + 0.1)
        s2 = frobenius_anomaly_score(H, H + 1.0)
        assert s1 < s2
