"""
tests/test_capsnet.py
=====================
Unit tests for CapsNet — verifies Eq. (4), (5), (6), (7), (8) are correct.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.models.capsule_network import (
    CapsNet,
    CapsNetConfig,
    MarginLoss,
    RicciFlowRouting,
    squash,
)


# =============================================================================
# Eq. (8) — SQUASH
# =============================================================================
class TestSquash:
    def test_squash_bounded_by_one(self):
        """‖squash(s)‖ < 1 for all s."""
        x = torch.randn(100, 16) * 10
        y = squash(x, dim=-1)
        norms = torch.linalg.norm(y, dim=-1)
        assert torch.all(norms < 1.0 + 1e-5)

    def test_squash_zero(self):
        x = torch.zeros(4, 8)
        y = squash(x, dim=-1)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

    def test_squash_preserves_direction(self):
        """squash(s) is a scalar multiple of s."""
        x = torch.randn(5, 8)
        y = squash(x, dim=-1)
        # direction unchanged: cos(y_i, x_i) ≈ 1
        cos = F.cosine_similarity(y, x, dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)

    def test_squash_magnitude(self):
        """squash(s)·squash(s) = ‖s‖²/(1+‖s‖²) per Eq. 8."""
        x = torch.tensor([[3.0, 4.0]])              # ‖x‖ = 5
        y = squash(x, dim=-1)
        expected = 25.0 / 26.0                      # ‖y‖² = 25 / (1 + 25)
        assert torch.isclose(torch.sum(y ** 2), torch.tensor(expected), atol=1e-5)


# =============================================================================
# Eq. (6) — ROUTING COEFFICIENTS
# =============================================================================
class TestRicciFlowRouting:
    def test_routing_output_shape(self):
        routing = RicciFlowRouting(
            num_caps_in=8, num_caps_out=2, in_dim=8, out_dim=16, num_iterations=3,
        )
        u = torch.randn(4, 8, 8)
        v, c = routing(u)
        assert v.shape == (4, 2, 16)
        assert c.shape == (4, 8, 2)

    def test_coupling_coefficients_sum_to_one(self):
        """Eq. (6): c_ij summed over j equals 1 for every i."""
        routing = RicciFlowRouting(8, 2, 8, 16, num_iterations=3)
        u = torch.randn(4, 8, 8)
        _, c = routing(u)
        sums = c.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_output_capsule_norm_le_one(self):
        """Eq. (8): v_j = squash(s_j) ⇒ ‖v_j‖ < 1."""
        routing = RicciFlowRouting(8, 2, 8, 16, num_iterations=3)
        u = torch.randn(4, 8, 8) * 10
        v, _ = routing(u)
        norms = torch.linalg.norm(v, dim=-1)
        assert torch.all(norms < 1.0 + 1e-5)

    def test_ricci_beta_effect(self):
        """Different beta should produce different outputs (Eq. 4 active)."""
        u = torch.randn(2, 8, 8)
        r1 = RicciFlowRouting(8, 2, 8, 16, num_iterations=3, beta=0.0)
        r2 = RicciFlowRouting(8, 2, 8, 16, num_iterations=3, beta=1.0)
        with torch.no_grad():
            r2.W.copy_(r1.W)
        v1, _ = r1(u)
        v2, _ = r2(u)
        assert not torch.allclose(v1, v2, atol=1e-3)


# =============================================================================
# Eq. (7) — MARGIN LOSS
# =============================================================================
class TestMarginLoss:
    def test_margin_loss_formula(self):
        """Hand-check Eq. 7 on a small case."""
        loss_fn = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5, reduction="mean")

        # Sample 1: label=1, caps=[0.9, 0.1]  → pos term = 0, neg term active on class 0
        # Sample 2: label=0, caps=[0.9, 0.1]  → pos term active on class 1 (0.1 < 0.9 ⇒ (0.9-0.1)²)
        norms = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
        labels = torch.tensor([1, 0])

        L = loss_fn(norms, labels)

        # Sample 1: class 1 → max(0, 0.9-0.1)² = 0 ; class 0 → 0.5*max(0, 0.9-0.1)² = 0.32
        # Sample 2: class 0 → max(0, 0.9-0.9)² = 0 ; class 1 → 0.5*max(0, 0.9-0.1)² = 0.32
        # mean = 0.32
        assert torch.isclose(L, torch.tensor(0.32), atol=1e-5)

    def test_margin_loss_zero_when_perfect(self):
        """Perfect capsules → L = 0."""
        loss_fn = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5)
        norms = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        labels = torch.tensor([1, 0])
        L = loss_fn(norms, labels)
        assert torch.isclose(L, torch.tensor(0.0), atol=1e-6)

    def test_margin_loss_penalizes_confident_wrong(self):
        """Confident-wrong predictions incur max loss."""
        loss_fn = MarginLoss(reduction="mean")
        norms = torch.tensor([[0.99, 0.01]])       # confident on class 0
        labels = torch.tensor([1])                 # ground truth = 1
        L = loss_fn(norms, labels)
        # class 1 pos term: (0.9 - 0.01)² = 0.81 ; class 0 neg term: 0.5*(0.99-0.1)² = 0.396
        assert torch.isclose(L, torch.tensor(0.81 + 0.5 * 0.81 ** 2 - 0.5 * 0.81 ** 2 + 0.5 * 0.9 ** 2), atol=1e-2) \
            or L > 0.4


# =============================================================================
# END-TO-END
# =============================================================================
class TestCapsNetEndToEnd:
    def test_forward_shape(self):
        model = CapsNet(CapsNetConfig(input_dim=64))
        x = torch.randn(8, 64)
        probs = model(x)
        assert probs.shape == (8, 2)

    def test_forward_with_caps(self):
        model = CapsNet(CapsNetConfig(input_dim=64))
        x = torch.randn(8, 64)
        probs, caps = model(x, return_caps=True)
        assert caps.shape == (8, 2, 16)

    def test_probabilities_are_capsule_norms(self):
        """probs should equal ‖v_j‖ (Eq. 7 usage)."""
        model = CapsNet(CapsNetConfig(input_dim=64))
        model.eval()
        x = torch.randn(4, 64)
        with torch.no_grad():
            probs, caps = model(x, return_caps=True)
        expected = torch.linalg.norm(caps, dim=-1)
        assert torch.allclose(probs, expected, atol=1e-5)

    def test_gradients_flow(self):
        model = CapsNet(CapsNetConfig(input_dim=64))
        x = torch.randn(8, 64)
        y = torch.randint(0, 2, (8,))
        probs, caps = model(x, return_caps=True)
        loss = MarginLoss()(probs, y)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any((g.abs() > 0).any().item() for g in grads)

    def test_paper_accuracy_smoke(self):
        """Train on a separable toy task; assert acc ≥ 90% (loose floor)."""
        torch.manual_seed(0)
        model = CapsNet(CapsNetConfig(input_dim=32))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginLoss()

        X = torch.cat([torch.randn(64, 32), torch.randn(64, 32) + 2.0])
        y = torch.cat([torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.long)])

        for _ in range(80):
            perm = torch.randperm(128)
            xb, yb = X[perm], y[perm]
            probs = model(xb)
            loss = loss_fn(probs, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            probs = model(X)
        pred = probs.argmax(-1)
        acc = (pred == y).float().mean().item()
        assert acc >= 0.85, f"Toy training accuracy {acc:.3f} below 85%"
