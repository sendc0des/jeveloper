"""
ModernBERT Backbone Wrapper.
Handles model loading, tokenizer management, special marker tokens ([OPT]),
and pooling operations (CLS pooling, marker pooling).
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from transformers import AutoModel, AutoTokenizer


class ModernBERTBackbone(nn.Module):
    """
    Backbone encoder wrapping answerdotai/ModernBERT-base.
    Supports native 8192 context, RoPE, and unpadded attention.
    """
    DEFAULT_MODEL_ID = "answerdotai/ModernBERT-base"
    OPTION_MARKER = "[OPT]"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Add special option marker token if not present
        if self.OPTION_MARKER not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.OPTION_MARKER]})
            self.option_marker_id = self.tokenizer.convert_tokens_to_ids(self.OPTION_MARKER)
        else:
            self.option_marker_id = self.tokenizer.convert_tokens_to_ids(self.OPTION_MARKER)

        # Load Model
        self.encoder = AutoModel.from_pretrained(model_id, dtype=self.dtype)
        
        # Resize token embeddings if special tokens were added
        if len(self.tokenizer) > self.encoder.config.vocab_size:
            self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.encoder.config.hidden_size  # 768 for ModernBERT-base
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through ModernBERT encoder.
        Returns: last_hidden_state of shape (batch_size, seq_len, hidden_size)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state

    def pool_cls(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Extracts representation of the [CLS] token (first token).
        Returns: (batch_size, hidden_size)
        """
        return last_hidden_state[:, 0, :]

    def pool_markers(
        self,
        last_hidden_state: torch.Tensor,
        marker_indices: List[List[int]]
    ) -> List[torch.Tensor]:
        """
        Extracts contextual representations at specific option marker indices.
        
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            marker_indices: List of lists, where marker_indices[b] contains the
                            token indices of options for item b in the batch.
        
        Returns:
            List of tensors [ (K_b, hidden_size) ] for each item in the batch.
        """
        batch_marker_embs = []
        for b, indices in enumerate(marker_indices):
            if len(indices) == 0:
                # Fallback to CLS if no markers provided
                batch_marker_embs.append(last_hidden_state[b, 0:1, :])
            else:
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=last_hidden_state.device)
                embs = torch.index_select(last_hidden_state[b], 0, idx_tensor)
                batch_marker_embs.append(embs)
        return batch_marker_embs
