import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    """
    Paper Eq. (7): L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1-T_k) * max(0, ||v_k|| - m-)^2
    """
    def __init__(self, m_plus: float = 0.9, m_minus: float = 0.1, lambda_: float = 0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, capsule_norms: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C = capsule_norms.shape
        T = F.one_hot(labels, num_classes=C).float()
        pos = torch.clamp(self.m_plus - capsule_norms, min=0.0) ** 2
        neg = torch.clamp(capsule_norms - self.m_minus, min=0.0) ** 2
        losses = T * pos + self.lambda_ * (1.0 - T) * neg
        return losses.sum(dim=-1).mean()

class NTXentLoss(nn.Module):
    """
    Paper Eq. (15): Contrastive Self-Supervised Learning Loss
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        pos = sim[torch.arange(2 * B), pos_idx]
        log_prob = pos - torch.logsumexp(sim, dim=-1)
        return -log_prob.mean()
