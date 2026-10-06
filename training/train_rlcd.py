"""
Phase 2: RLCD (Reinforcement Learning for Calibrated Decisions) Training Pipeline.
Optimizes model confidence using strictly proper scoring rules (Brier score),
Expected Calibration Error (ECE) penalties, and reference policy regularization.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List, Dict, Any, Optional

from core.unified_model import JevModel
from data.formatter import DecisionFormatter
from data.dataset_builder import create_decision_dataloader
from calibration.proper_scoring import RLCDCalibrationLoss
from calibration.metrics import compute_ece, compute_brier_score
from training.config import TrainingConfig


def train_rlcd(
    train_records: List[Dict[str, Any]],
    sft_checkpoint_dir: str = "checkpoints/sft_model",
    output_dir: str = "checkpoints/rlcd_model",
    config: Optional[TrainingConfig] = None
):
    cfg = config or TrainingConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting Phase 2 (RLCD Calibration) on device: {device}")

    # 1. Load Pre-Aligned SFT Model
    model = JevModel(
        model_id=sft_checkpoint_dir if os.path.exists(os.path.join(sft_checkpoint_dir, "config.json")) else cfg.model_id,
        device=device,
        num_score_bins=cfg.num_score_bins,
        dropout=cfg.dropout
    )
    if os.path.exists(os.path.join(sft_checkpoint_dir, "jev_heads.pt")):
        model.load_heads(sft_checkpoint_dir)
    model.train()

    # 2. Formatter & DataLoader
    formatter = DecisionFormatter(model.tokenizer)
    train_loader = create_decision_dataloader(
        records=train_records,
        formatter=formatter,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    # 3. RLCD Loss Function
    rlcd_loss_fn = RLCDCalibrationLoss(
        brier_weight=cfg.brier_weight,
        ce_weight=cfg.ce_weight,
        ece_weight=cfg.ece_penalty_weight
    )

    # 4. Optimizer (Fine-tuning decision heads & higher transformer layers)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=cfg.weight_decay)
    use_cuda_amp = (device == "cuda" and cfg.fp16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

    print(f"[*] RLCD Training: Optimizing for calibrated confidence & proper scoring rules.")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_brier = 0.0
        total_cal_gap = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_indices = batch["marker_indices"]
            task_types = batch["task_types"]
            targets = batch["targets"]

            with torch.amp.autocast('cuda', enabled=use_cuda_amp):
                batch_losses = []

                for i, ttype in enumerate(task_types):
                    sub_input_ids = input_ids[i:i+1]
                    sub_mask = attention_mask[i:i+1]

                    if ttype == "choice":
                        res = model.forward_choice(
                            input_ids=sub_input_ids,
                            attention_mask=sub_mask,
                            marker_indices=[marker_indices[i]]
                        )
                        logits = res["logits"][0].unsqueeze(0)  # (1, K)
                        probs = res["probabilities"][0].unsqueeze(0)  # (1, K)
                        target_idx = torch.tensor([targets[i]], dtype=torch.long, device=device)

                        loss_dict = rlcd_loss_fn(logits=logits, probs=probs, targets=target_idx)
                        batch_losses.append(loss_dict["loss"])
                        total_brier += loss_dict["brier_loss"].item()
                        total_cal_gap += loss_dict["calibration_gap"].item()

                    elif ttype == "noul":
                        res = model.forward_noul(
                            input_ids=sub_input_ids,
                            attention_mask=sub_mask
                        )
                        p_true = res["probability"]
                        probs_2d = torch.stack([1.0 - p_true, p_true], dim=-1)  # (1, 2)
                        logits_2d = torch.stack([-res["logits"], res["logits"]], dim=-1)
                        target_idx = torch.tensor([int(targets[i])], dtype=torch.long, device=device)

                        loss_dict = rlcd_loss_fn(logits=logits_2d, probs=probs_2d, targets=target_idx)
                        batch_losses.append(loss_dict["loss"])

                if batch_losses:
                    loss = torch.stack(batch_losses).mean() / cfg.gradient_accumulation_steps
                else:
                    continue

            scaler.scale(loss).backward()
            total_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"[*] RLCD Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f} (Brier Penalty Active)")

    print(f"[*] RLCD Calibration complete.")
    model.save_checkpoint(output_dir)
    return model
