"""
Pytest smoke test for ModernBERT loading and forward pass.
"""
import torch
import pytest
from transformers import AutoTokenizer, AutoModel

def test_modernbert_forward():
    model_id = "answerdotai/ModernBERT-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    
    prompt = "[CLS] State: Test system state. [SEP] Question: Is this working? [SEP]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    assert outputs.last_hidden_state is not None
    assert outputs.last_hidden_state.shape[-1] == 768
    assert outputs.last_hidden_state.shape[0] == 1
