"""
Training and Hardware Optimization Configuration for Jevloper.
Budgeted specifically for RTX 4050 Laptop GPU (6GB VRAM) with mixed precision.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # Model
    model_id: str = "answerdotai/ModernBERT-base"
    num_score_bins: int = 10
    dropout: float = 0.1

    # Hardware & Memory Optimization
    device: str = "cuda"
    fp16: bool = True
    batch_size: int = 8                    # Safe for 6GB VRAM
    gradient_accumulation_steps: int = 2  # Effective batch size = 16
    max_seq_length: int = 512

    # Optimization
    backbone_lr: float = 3e-5              # Gentle fine-tuning for pre-trained weights
    head_lr: float = 1e-4                  # Faster learning for new decision heads
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    epochs: int = 3

    # RLCD Calibration Weights
    brier_weight: float = 0.5
    ce_weight: float = 0.5
    ece_penalty_weight: float = 0.2

    # Logging & Checkpointing
    eval_steps: int = 50
    save_dir: str = "checkpoints"
    seed: int = 42
