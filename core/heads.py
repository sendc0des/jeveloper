"""
Specialized Decision Heads for Jev Architecture.
Includes:
- DynamicChoiceHead: Single-pass scoring of arbitrary candidate options.
- DistributionalScoreHead: Bounded continuous / ordinal scoring with uncertainty.
- NoulHead: Binary hypothesis validation with calibrated confidence.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional


class DynamicChoiceHead(nn.Module):
    """
    Dynamic Choice Head.
    Scores an arbitrary number of candidate options (from 2 up to 255)
    by projecting each option's contextual token embedding into a scalar decision logit.
    """
    def __init__(self, hidden_size: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # Learnable temperature parameter for calibration (initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(
        self,
        option_embeddings_batch: List[torch.Tensor],
        temperature_override: Optional[float] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            option_embeddings_batch: List of tensors of shape (K_b, hidden_size),
                                     where K_b is the number of options for batch item b.
            temperature_override: Optional explicit temperature value.

        Returns:
            logits_list: List of 1D tensors [ (K_b,) ] of raw logits
            probs_list:  List of 1D tensors [ (K_b,) ] of normalized probabilities
        """
        temp = temperature_override if temperature_override is not None else torch.clamp(self.temperature, min=0.01)
        dtype = next(self.scorer.parameters()).dtype
        
        logits_list = []
        probs_list = []

        for embs in option_embeddings_batch:
            embs = embs.to(dtype)
            # embs shape: (K, hidden_size)
            logits = self.scorer(embs).squeeze(-1)  # shape: (K,)
            probs = torch.softmax(logits / temp, dim=-1)
            logits_list.append(logits)
            probs_list.append(probs)

        return logits_list, probs_list


class DistributionalScoreHead(nn.Module):
    """
    Distributional Score Head for continuous / ordinal evaluation.
    Discretizes bounded range [min_score, max_score] into B bins.
    Predicts probability mass over bins, computing expected value and variance.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_bins: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_bins = num_bins
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(
        self,
        cls_embedding: torch.Tensor,
        min_score: float = 1.0,
        max_score: float = 5.0,
        temperature_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        temp = temperature_override if temperature_override is not None else torch.clamp(self.temperature, min=0.01)
        dtype = next(self.mlp.parameters()).dtype
        cls_embedding = cls_embedding.to(dtype)
        logits = self.mlp(cls_embedding)  # (batch_size, num_bins)
        probs = torch.softmax(logits / temp, dim=-1)

        device = cls_embedding.device
        # Compute bin center values
        step = (max_score - min_score) / self.num_bins
        bin_centers = torch.tensor(
            [min_score + (i + 0.5) * step for i in range(self.num_bins)],
            dtype=dtype,
            device=device
        )  # (num_bins,)

        # Expected score: sum(probs * bin_centers)
        expected_score = torch.sum(probs * bin_centers, dim=-1)  # (batch_size,)

        # Variance: sum(probs * (bin_centers - expected_score)^2)
        diff = bin_centers.unsqueeze(0) - expected_score.unsqueeze(-1)  # (batch_size, num_bins)
        variance = torch.sum(probs * (diff ** 2), dim=-1)  # (batch_size,)

        return {
            "logits": logits,
            "probs": probs,
            "expected_score": expected_score,
            "variance": variance,
            "bin_centers": bin_centers
        }


class NoulHead(nn.Module):
    """
    Noul Head for binary hypothesis verification (True / False).
    Outputs calibrated probability of truth and decision confidence.
    """
    def __init__(self, hidden_size: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(
        self,
        cls_embedding: torch.Tensor,
        temperature_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        temp = temperature_override if temperature_override is not None else torch.clamp(self.temperature, min=0.01)
        dtype = next(self.classifier.parameters()).dtype
        cls_embedding = cls_embedding.to(dtype)
        temp = temperature_override if temperature_override is not None else torch.clamp(self.temperature, min=0.01)
        logits = self.classifier(cls_embedding).squeeze(-1)  # (batch_size,)
        prob_true = torch.sigmoid(logits / temp)
        certainty = torch.maximum(prob_true, 1.0 - prob_true)

        return {
            "logits": logits,
            "probability": prob_true,
            "confidence": certainty
        }
