"""
Unified Jev Model Architecture.
Combines ModernBERT backbone with multi-task decision heads:
- Dynamic Choice (single-pass dynamic candidates)
- Distributional Score (expected value & variance)
- Noul (binary hypothesis verification)
"""
import os
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union

from core.backbone import ModernBERTBackbone
from core.heads import DynamicChoiceHead, DistributionalScoreHead, NoulHead


class JevModel(nn.Module):
    """
    Unified local decision engine model.
    """
    def __init__(
        self,
        model_id: str = ModernBERTBackbone.DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        num_score_bins: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or (torch.float16 if self.device_name == "cuda" else torch.float32)

        # Shared Backbone Encoder
        self.backbone = ModernBERTBackbone(
            model_id=model_id,
            device=self.device_name,
            torch_dtype=self.dtype
        )
        hidden_size = self.backbone.hidden_size

        # Multi-Task Decision Heads
        self.choice_head = DynamicChoiceHead(hidden_size=hidden_size, dropout=dropout)
        self.score_head = DistributionalScoreHead(hidden_size=hidden_size, num_bins=num_score_bins, dropout=dropout)
        self.noul_head = NoulHead(hidden_size=hidden_size, dropout=dropout)

        self.to(device=self.device_name, dtype=self.dtype)

    @property
    def tokenizer(self):
        return self.backbone.tokenizer

    def forward_choice(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_indices: List[List[int]],
        temperature_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for categorical Choice decisions.
        """
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        option_embs = self.backbone.pool_markers(hidden, marker_indices)
        logits_list, probs_list = self.choice_head(option_embs, temperature_override=temperature_override)
        return {
            "logits": logits_list,
            "probabilities": probs_list
        }

    def forward_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        min_score: float = 1.0,
        max_score: float = 5.0,
        temperature_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for continuous / ordinal Score evaluations.
        """
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = self.backbone.pool_cls(hidden)
        return self.score_head(
            cls_emb,
            min_score=min_score,
            max_score=max_score,
            temperature_override=temperature_override
        )

    def forward_noul(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for binary Noul hypothesis checks.
        """
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = self.backbone.pool_cls(hidden)
        return self.noul_head(cls_emb, temperature_override=temperature_override)

    def save_checkpoint(self, save_dir: str):
        """
        Saves full model weights and tokenizer to directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save backbone encoder and tokenizer
        self.backbone.encoder.save_pretrained(save_dir)
        self.backbone.tokenizer.save_pretrained(save_dir)
        
        # Save heads state dict
        heads_state = {
            "choice_head": self.choice_head.state_dict(),
            "score_head": self.score_head.state_dict(),
            "noul_head": self.noul_head.state_dict()
        }
        torch.save(heads_state, os.path.join(save_dir, "jev_heads.pt"))
        print(f"[*] Saved JevModel checkpoint to {save_dir}")

    def load_heads(self, checkpoint_path: str):
        """
        Loads heads state dict from checkpoint.
        """
        heads_path = checkpoint_path if os.path.isfile(checkpoint_path) else os.path.join(checkpoint_path, "jev_heads.pt")
        if os.path.exists(heads_path):
            state = torch.load(heads_path, map_location=self.device_name)
            self.choice_head.load_state_dict(state["choice_head"])
            self.score_head.load_state_dict(state["score_head"])
            self.noul_head.load_state_dict(state["noul_head"])
            print(f"[*] Loaded Jev heads from {heads_path}")
        else:
            raise FileNotFoundError(f"Heads file not found at {heads_path}")
