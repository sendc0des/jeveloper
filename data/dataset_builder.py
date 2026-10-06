"""
Unified PyTorch Dataset and DataLoader with dynamic collate batching for Jev.
Supports heterogeneous batches containing Choice, Score, and Noul tasks.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from data.formatter import DecisionFormatter


class MultiTaskDecisionDataset(Dataset):
    """
    Multi-task Dataset for Jev decision engine.
    """
    def __init__(self, records: List[Dict[str, Any]], formatter: DecisionFormatter):
        self.records = records
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        task_type = item["task_type"]

        if task_type == "choice":
            formatted = self.formatter.format_choice(
                state=item["state"],
                question=item["question"],
                options=item["options"]
            )
            return {
                "task_type": "choice",
                "input_ids": formatted["input_ids"],
                "attention_mask": formatted["attention_mask"],
                "marker_indices": formatted["marker_indices"],
                "target": item.get("target_idx", 0),
                "options": formatted["option_ids"]
            }

        elif task_type == "score":
            formatted = self.formatter.format_score(
                state=item["state"],
                question=item["question"]
            )
            return {
                "task_type": "score",
                "input_ids": formatted["input_ids"],
                "attention_mask": formatted["attention_mask"],
                "marker_indices": [],
                "target": float(item.get("target_score", 3.0)),
                "min_score": float(item.get("min_score", 1.0)),
                "max_score": float(item.get("max_score", 5.0))
            }

        elif task_type == "noul":
            formatted = self.formatter.format_noul(
                state=item["state"],
                hypothesis=item["hypothesis"]
            )
            return {
                "task_type": "noul",
                "input_ids": formatted["input_ids"],
                "attention_mask": formatted["attention_mask"],
                "marker_indices": [],
                "target": 1.0 if item.get("target_bool", False) else 0.0
            }

        raise ValueError(f"Unknown task_type: {task_type}")


def dynamic_decision_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Dynamic collate function that pads sequences to the maximum length in the batch.
    """
    pad_token_id = 0
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    marker_indices_list = []
    task_types = []
    targets = []
    metadata = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        # Right-pad input_ids
        padded_ids = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_mask = torch.cat([item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)])

        batch_input_ids.append(padded_ids)
        batch_attention_mask.append(padded_mask)
        marker_indices_list.append(item["marker_indices"])
        task_types.append(item["task_type"])
        targets.append(item["target"])
        metadata.append(item)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "marker_indices": marker_indices_list,
        "task_types": task_types,
        "targets": targets,
        "metadata": metadata
    }


def create_decision_dataloader(
    records: List[Dict[str, Any]],
    formatter: DecisionFormatter,
    batch_size: int = 16,
    shuffle: bool = True
) -> DataLoader:
    """
    Factory helper to instantiate a DataLoader with dynamic collate.
    """
    dataset = MultiTaskDecisionDataset(records, formatter)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dynamic_decision_collate
    )
