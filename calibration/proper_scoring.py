"""
Strictly Proper Scoring Rules and Calibration Loss Functions for RLCD.
Implements:
- Multi-class Brier Loss
- Binary Brier Loss
- Calibrated Multi-Task Loss with ECE / entropy regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BrierScoreLoss(nn.Module):
    """
    Multi-class Brier score loss.
    L_Brier = mean_i sum_k (p_ik - y_ik)^2
    Unlike standard cross-entropy, Brier score is a strictly proper scoring rule
    that directly penalizes miscalibrated probability distributions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        probs: (N, K) normalized probabilities
        targets: (N,) target class indices
        """
        N, K = probs.shape
        one_hot = F.one_hot(targets, num_classes=K).to(probs.dtype)
        # Brier score: mean squared error between probabilities and one-hot vectors
        return torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))


class BinaryBrierScoreLoss(nn.Module):
    """
    Binary Brier score loss for Noul tasks.
    """
    def __init__(self):
        super().__init__()

    def forward(self, prob_true: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        prob_true: (N,) probability in [0, 1]
        targets: (N,) float targets {0.0, 1.0}
        """
        targets = targets.to(prob_true.dtype)
        return torch.mean((prob_true - targets) ** 2)


class RLCDCalibrationLoss(nn.Module):
    """
    Composite RLCD training loss.
    Combines:
    1. Cross-entropy task alignment loss (L_CE)
    2. Strictly proper scoring Brier loss (L_Brier)
    3. Soft Expected Calibration Error (ECE) penalty
    """
    def __init__(
        self,
        brier_weight: float = 0.5,
        ce_weight: float = 0.5,
        ece_weight: float = 0.2
    ):
        super().__init__()
        self.brier_weight = brier_weight
        self.ce_weight = ce_weight
        self.ece_weight = ece_weight
        self.brier_loss = BrierScoreLoss()

    def forward(self, logits: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        logits: (N, K) raw unnormalized logits
        probs: (N, K) softmax probabilities
        targets: (N,) ground truth target indices
        """
        ce_loss = F.cross_entropy(logits, targets)
        brier_loss = self.brier_loss(probs, targets)

        # Soft ECE penalty (differentiable approximation using confidence vs correctness)
        confidences, preds = torch.max(probs, dim=-1)
        accuracies = (preds == targets).to(probs.dtype)
        calibration_gap = torch.mean(torch.abs(confidences - accuracies))

        total_loss = (
            self.ce_weight * ce_loss +
            self.brier_weight * brier_loss +
            self.ece_weight * calibration_gap
        )

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "brier_loss": brier_loss,
            "calibration_gap": calibration_gap
        }
