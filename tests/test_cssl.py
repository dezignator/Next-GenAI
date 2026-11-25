"""
tests/test_cssl.py
==================
Unit tests for CSSL — verifies Eq. (15) and augmentation invariants.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.cssl import (
    CSSL,
    CSSLConfig,
    CSSLEncoder,
    DoHAugmenter,
    MemoryBank,
    ProjectionHead,
    nt_xent_loss,
)


class TestDoHAugmenter:
    def test_preserves_shape(self):
        aug = DoHAugmenter()
        x = torch.randn(8, 64)
        y = aug(x)
        assert y.shape == x.shape

    def test_output_differs_from_input(self):
        aug = DoHAugmenter(noise_std=0.1)
        x = torch.ones(8, 64)
        y = aug(x)
        assert not torch.allclose(x, y)

    def test_noise_scale_small(self):
        """With small noise, augmentations stay near original."""
        aug = DoHAugmenter(noise_std=0.001, dropout_p=0.0)
        x = torch.randn(8, 64)
        y = aug(x)
        assert (x - y).abs().mean() < 0.1


class TestProjectionHead:
    def test_output_normalized(self):
        head = ProjectionHead(in_dim=64, hidden=128, out_dim=32)
        x = torch.randn(8, 64)
        z = head(x)
        norms = torch.linalg.norm(z, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestNTXentLoss:
    def test_perfectly_aligned_views_low_loss(self):
        """If z1 == z2 per row, loss should be near log(2B-1) minimum."""
        z = F.normalize(torch.randn(16, 32), dim=-1)
        loss = nt_xent_loss(z, z, temperature=0.07)
        # Perfect alignment → numerator large, so loss near 0 for each sample
        assert loss < 1.0

    def test_misaligned_high_loss(self):
        """Random views → loss near log(2B-1)."""
        z1 = F.normalize(torch.randn(32, 32), dim=-1)
        z2 = F.normalize(torch.randn(32, 32), dim=-1)
        loss = nt_xent_loss(z1, z2, temperature=0.07)
        import math
        expected = math.log(2 * 32 - 1)
        assert loss > expected * 0.5           # in the right order of magnitude

    def test_temperature_effect(self):
        """Lower τ → sharper distribution → higher gradient sensitivity."""
        z1 = F.normalize(torch.randn(16, 32), dim=-1)
        z2 = F.normalize(torch.randn(16, 32), dim=-1)
        loss_hi = nt_xent_loss(z1, z2, temperature=1.0)
        loss_lo = nt_xent_loss(z1, z2, temperature=0.07)
        assert loss_lo != loss_hi


class TestCSSL:
    def test_forward_returns_scalar(self):
        model = CSSL(CSSLConfig(encoder_in_dim=32))
        x = torch.randn(16, 32)
        loss, z = model(x)
        assert loss.dim() == 0
        assert z.shape == (16, 128)           # projection_dim

    def test_gradients_flow(self):
        model = CSSL(CSSLConfig(encoder_in_dim=32))
        x = torch.randn(16, 32)
        loss, _ = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any((g.abs() > 0).any().item() for g in grads)

    def test_embedding_shape(self):
        model = CSSL(CSSLConfig(encoder_in_dim=32))
        x = torch.randn(8, 32)
        emb = model.get_embedding(x)
        assert emb.shape == (8, 128)


class TestMemoryBank:
    def test_update_and_sample(self):
        bank = MemoryBank(size=32, dim=16)
        z = F.normalize(torch.randn(8, 16), dim=-1)
        bank.update(z)
        sample = bank.sample(4)
        assert sample.shape == (4, 16)
