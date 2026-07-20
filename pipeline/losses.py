import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return bce + (1.0 - dice.mean())


class DistillSegLoss(nn.Module):
    """loss = alpha * DiceBCELoss(student, GT) + (1-alpha) * BCE(student, teacher_prob)
    -- same fusion formula used everywhere else distillation appears in this project."""

    def __init__(self, alpha: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.sup_loss = DiceBCELoss()

    def forward(self, logits: torch.Tensor, gt: torch.Tensor, teacher_prob: torch.Tensor) -> torch.Tensor:
        sup = self.sup_loss(logits, gt)
        soft = F.binary_cross_entropy_with_logits(logits, teacher_prob)
        return self.alpha * sup + (1.0 - self.alpha) * soft
