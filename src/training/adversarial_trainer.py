import torch
import torch.nn as nn
from loguru import logger

class AdversarialTrainer:
    """
    Implements FGSM (Fast Gradient Sign Method) for adversarial training.
    Paper Sec. 3.2: "model received regular injections of adverse samples during training"
    """
    def __init__(self, model: nn.Module, epsilon: float = 0.01):
        self.model = model
        self.epsilon = epsilon

    def generate_fgsm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generates adversarial examples via FGSM."""
        x_adv = x.clone().detach().requires_grad_(True)
        logits = self.model(x_adv)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        loss = nn.CrossEntropyLoss()(logits, y)
        self.model.zero_grad()
        loss.backward()
        
        # Apply perturbation
        with torch.no_grad():
            x_adv = x_adv + self.epsilon * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1) # Assuming normalized input
            
        return x_adv.detach()

    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Single training step mixing clean and adversarial data."""
        self.model.train()
        
        # 1. Generate Adversarial
        x_adv = self.generate_fgsm(x, y)
        
        # 2. Mix Clean and Adversarial (50/50)
        x_mix = torch.cat([x, x_adv], dim=0)
        y_mix = torch.cat([y, y], dim=0)
        
        # 3. Train on mixed batch
        optimizer.zero_grad()
        logits = self.model(x_mix)
        if isinstance(logits, tuple):
            logits = logits[0]
            
        loss = nn.CrossEntropyLoss()(logits, y_mix)
        loss.backward()
        optimizer.step()
        
        return loss.item()
