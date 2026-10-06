"""
Phase 1: Supervised Fine-Tuning (SFT) Training Pipeline.
Multi-task supervised training across Choice, Score, and Noul tasks.
Optimized for RTX 4050 (6GB VRAM) using mixed precision and gradient accumulation.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict, Any, Optional

from core.unified_model import JevModel
from data.formatter import DecisionFormatter
from data.dataset_builder import create_decision_dataloader
from training.config import TrainingConfig


def train_sft(
    train_records: List[Dict[str, Any]],
    val_records: Optional[List[Dict[str, Any]]] = None,
    config: Optional[TrainingConfig] = None,
    output_dir: str = "checkpoints/sft_model"
):
    cfg = config or TrainingConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting Phase 1 (SFT) on device: {device}")

    # 1. Initialize Model
    model = JevModel(
        model_id=cfg.model_id,
        device=device,
        num_score_bins=cfg.num_score_bins,
        dropout=cfg.dropout
    )
    model.train()

    # 2. Formatter & DataLoader
    formatter = DecisionFormatter(model.tokenizer)
    train_loader = create_decision_dataloader(
        records=train_records,
        formatter=formatter,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    # 3. Parameter Groups (different learning rates for backbone vs heads)
    backbone_params = list(model.backbone.parameters())
    heads_params = (
        list(model.choice_head.parameters()) +
        list(model.score_head.parameters()) +
        list(model.noul_head.parameters())
    )
    optimizer = AdamW([
        {"params": backbone_params, "lr": cfg.backbone_lr, "weight_decay": cfg.weight_decay},
        {"params": heads_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay}
    ])

    total_steps = (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    use_cuda_amp = (device == "cuda" and cfg.fp16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    print(f"[*] Total training samples: {len(train_records)} | Batches per epoch: {len(train_loader)}")
    print(f"[*] Effective batch size: {cfg.batch_size * cfg.gradient_accumulation_steps}")

    global_step = 0
    t_start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_indices = batch["marker_indices"]
            task_types = batch["task_types"]
            targets = batch["targets"]

            with torch.amp.autocast('cuda', enabled=use_cuda_amp):
                batch_losses = []

                # Group by task type within the batch for clean forward calls
                for i, ttype in enumerate(task_types):
                    sub_input_ids = input_ids[i:i+1]
                    sub_mask = attention_mask[i:i+1]

                    if ttype == "choice":
                        res = model.forward_choice(
                            input_ids=sub_input_ids,
                            attention_mask=sub_mask,
                            marker_indices=[marker_indices[i]]
                        )
                        logits = res["logits"][0]  # (K,)
                        target_idx = torch.tensor(targets[i], dtype=torch.long, device=device)
                        loss = ce_loss(logits.unsqueeze(0), target_idx.unsqueeze(0))
                        batch_losses.append(loss)

                    elif ttype == "score":
                        meta = batch["metadata"][i]
                        res = model.forward_score(
                            input_ids=sub_input_ids,
                            attention_mask=sub_mask,
                            min_score=meta.get("min_score", 1.0),
                            max_score=meta.get("max_score", 5.0)
                        )
                        # Discretize continuous target score into nearest bin
                        target_val = float(targets[i])
                        bin_centers = res["bin_centers"]
                        nearest_bin = int(torch.argmin(torch.abs(bin_centers - target_val)).item())
                        loss = ce_loss(res["logits"], torch.tensor([nearest_bin], device=device))
                        batch_losses.append(loss)

                    elif ttype == "noul":
                        res = model.forward_noul(
                            input_ids=sub_input_ids,
                            attention_mask=sub_mask
                        )
                        target_val = torch.tensor([float(targets[i])], dtype=torch.float, device=device)
                        loss = bce_loss(res["logits"].unsqueeze(0), target_val)
                        batch_losses.append(loss)

                if batch_losses:
                    loss = torch.stack(batch_losses).mean() / cfg.gradient_accumulation_steps
                else:
                    continue

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % 20 == 0:
                    avg_loss = running_loss / (step + 1)
                    print(f"  [Epoch {epoch+1}/{cfg.epochs} | Step {step+1}] Loss: {avg_loss:.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"[*] Epoch {epoch+1} Complete. Average Loss: {epoch_loss:.4f}")

    total_time = time.time() - t_start
    print(f"[*] Training finished in {total_time:.1f}s.")
    model.save_checkpoint(output_dir)
    return model
