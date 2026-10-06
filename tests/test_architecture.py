"""
Unit tests for core architecture, backbone, and decision heads.
"""
import torch
import pytest
from core.backbone import ModernBERTBackbone
from core.heads import DynamicChoiceHead, DistributionalScoreHead, NoulHead
from core.unified_model import JevModel


def test_heads_shapes_and_gradients():
    hidden_size = 768
    batch_size = 2

    # 1. Test Choice Head
    choice_head = DynamicChoiceHead(hidden_size=hidden_size)
    # Batch item 0 has 3 choices, item 1 has 5 choices
    embs_batch = [
        torch.randn(3, hidden_size, requires_grad=True),
        torch.randn(5, hidden_size, requires_grad=True)
    ]
    logits, probs = choice_head(embs_batch)
    assert len(logits) == 2
    assert logits[0].shape == (3,)
    assert logits[1].shape == (5,)
    assert torch.allclose(probs[0].sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(probs[1].sum(), torch.tensor(1.0), atol=1e-4)

    # Gradient flow test
    loss = logits[0].sum() + logits[1].sum()
    loss.backward()
    assert embs_batch[0].grad is not None

    # 2. Test Score Head
    score_head = DistributionalScoreHead(hidden_size=hidden_size, num_bins=10)
    cls_embs = torch.randn(batch_size, hidden_size, requires_grad=True)
    score_out = score_head(cls_embs, min_score=1.0, max_score=5.0)
    assert score_out["expected_score"].shape == (batch_size,)
    assert score_out["variance"].shape == (batch_size,)
    assert score_out["probs"].shape == (batch_size, 10)
    assert torch.all(score_out["expected_score"] >= 1.0)
    assert torch.all(score_out["expected_score"] <= 5.0)

    # 3. Test Noul Head
    noul_head = NoulHead(hidden_size=hidden_size)
    noul_out = noul_head(cls_embs)
    assert noul_out["probability"].shape == (batch_size,)
    assert torch.all(noul_out["probability"] >= 0.0)
    assert torch.all(noul_out["probability"] <= 1.0)
    assert torch.all(noul_out["confidence"] >= 0.5)


def test_unified_model_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = JevModel(device=device, torch_dtype=dtype)
    model.eval()

    # Create synthetic tokens
    input_ids = torch.randint(10, 1000, (1, 16), device=device)
    attention_mask = torch.ones((1, 16), dtype=torch.long, device=device)

    # Choice forward with markers at indices 4, 8, 12
    choice_res = model.forward_choice(
        input_ids=input_ids,
        attention_mask=attention_mask,
        marker_indices=[[4, 8, 12]]
    )
    assert len(choice_res["probabilities"][0]) == 3

    # Score forward
    score_res = model.forward_score(
        input_ids=input_ids,
        attention_mask=attention_mask,
        min_score=1.0,
        max_score=5.0
    )
    assert score_res["expected_score"].shape == (1,)

    # Noul forward
    noul_res = model.forward_noul(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    assert noul_res["probability"].shape == (1,)
