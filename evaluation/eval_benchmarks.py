"""
Enterprise Evaluation Suite for Jev Decision Engine.
Computes Accuracy, Macro-F1, Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and Brier Score across tasks.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score

from core.unified_model import JevModel
from data.formatter import DecisionFormatter
from calibration.metrics import compute_ece, compute_brier_score, compute_binary_brier_score, compute_nll


@torch.no_grad()
def evaluate_choice_dataset(
    model: JevModel,
    records: List[Dict[str, Any]],
    formatter: Optional[DecisionFormatter] = None
) -> Dict[str, Any]:
    """
    Evaluates choice dataset (e.g. Banking77 / AG News).
    """
    model.eval()
    fmt = formatter or DecisionFormatter(model.tokenizer)
    device = model.device_name

    all_preds = []
    all_targets = []
    all_confs = []
    all_probs = []

    for item in records:
        formatted = fmt.format_choice(
            state=item["state"],
            question=item["question"],
            options=item["options"]
        )
        input_ids = formatted["input_ids"].unsqueeze(0).to(device)
        attention_mask = formatted["attention_mask"].unsqueeze(0).to(device)
        marker_indices = [formatted["marker_indices"]]

        outputs = model.forward_choice(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_indices=marker_indices
        )

        probs = outputs["probabilities"][0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        target_idx = item["target_idx"]

        all_preds.append(pred_idx)
        all_targets.append(target_idx)
        all_confs.append(conf)
        all_probs.append(probs)

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)
    confs_np = np.array(all_confs)

    acc = float(accuracy_score(targets_np, preds_np))
    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))
    ece_data = compute_ece(confidences=confs_np, predictions=preds_np, targets=targets_np, num_bins=10)

    # Pad probability matrix for Brier calculation if variable choices
    max_k = max(len(p) for p in all_probs)
    padded_probs = np.zeros((len(all_probs), max_k), dtype=np.float32)
    for i, p in enumerate(all_probs):
        padded_probs[i, :len(p)] = p
    brier = compute_brier_score(padded_probs, targets_np)

    return {
        "task": "choice",
        "sample_count": len(records),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece": ece_data["ece"],
        "mce": ece_data["mce"],
        "brier_score": brier,
        "bins_data": ece_data["bins_data"]
    }


@torch.no_grad()
def evaluate_noul_dataset(
    model: JevModel,
    records: List[Dict[str, Any]],
    formatter: Optional[DecisionFormatter] = None
) -> Dict[str, Any]:
    """
    Evaluates binary Noul dataset (e.g. BoolQ).
    """
    model.eval()
    fmt = formatter or DecisionFormatter(model.tokenizer)
    device = model.device_name

    all_preds = []
    all_targets = []
    all_confs = []
    all_probs = []

    for item in records:
        formatted = fmt.format_noul(
            state=item["state"],
            hypothesis=item["hypothesis"]
        )
        input_ids = formatted["input_ids"].unsqueeze(0).to(device)
        attention_mask = formatted["attention_mask"].unsqueeze(0).to(device)

        outputs = model.forward_noul(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        p_true = float(outputs["probability"][0].cpu().item())
        pred_bool = p_true >= 0.5
        conf = float(outputs["confidence"][0].cpu().item())
        target_bool = bool(item["target_bool"])

        all_preds.append(int(pred_bool))
        all_targets.append(int(target_bool))
        all_confs.append(conf)
        all_probs.append(p_true)

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)
    confs_np = np.array(all_confs)
    probs_np = np.array(all_probs)

    acc = float(accuracy_score(targets_np, preds_np))
    f1 = float(f1_score(targets_np, preds_np, average="binary", zero_division=0))
    ece_data = compute_ece(confidences=confs_np, predictions=preds_np, targets=targets_np, num_bins=10)
    brier = compute_binary_brier_score(probs_np, targets_np)

    return {
        "task": "noul",
        "sample_count": len(records),
        "accuracy": acc,
        "f1": f1,
        "ece": ece_data["ece"],
        "mce": ece_data["mce"],
        "brier_score": brier,
        "bins_data": ece_data["bins_data"]
    }
