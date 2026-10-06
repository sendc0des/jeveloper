"""
Post-hoc Temperature Scaling Optimizer.
Learns a single scalar temperature T > 0 on a calibration/validation set
to rescale logits and minimize Expected Calibration Error without affecting accuracy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional
import numpy as np


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling layer.
    p = softmax(logits / T)
    """
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scales logits by 1/T.
        """
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp

    def fit(
        self,
        valid_logits: torch.Tensor,
        valid_targets: torch.Tensor,
        max_iters: int = 100,
        lr: float = 0.01
    ) -> float:
        """
        Fits temperature parameter using L-BFGS or Adam to minimize Cross-Entropy (NLL)
        on the validation set.
        """
        device = valid_logits.device
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(valid_logits)
            loss = criterion(scaled_logits, valid_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        final_temp = float(torch.clamp(self.temperature, min=0.01).item())
        print(f"[*] Optimized temperature parameter: T = {final_temp:.4f}")
        return final_temp
