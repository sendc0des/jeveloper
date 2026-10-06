"""
Dataset Download and Preparation Utility.
Downloads and pre-processes standard benchmark datasets for Choice, Score, and Noul tasks:
- Banking77: 77 Customer support intents (Choice)
- AG News: 4 News categorization topics (Choice)
- SST-5 / SST-2: Fine-grained sentiment ratings 1-5 (Score)
- BoolQ: Boolean QA Yes/No validation (Noul)
"""
import os
import json
from typing import Dict, List, Any
from datasets import load_dataset


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def download_banking77(sample_limit: int = 2000) -> str:
    """
    Downloads Banking77 dataset for Choice / Routing evaluation.
    """
    out_file = os.path.join(CACHE_DIR, "banking77.json")
    if os.path.exists(out_file):
        print(f"[*] Banking77 already cached at {out_file}")
        return out_file

    print("[*] Downloading Banking77 from Hugging Face Hub...")
    ds = load_dataset("PolyAI/banking77", split="train")
    label_names = ds.features["label"].names

    records = []
    for item in ds.select(range(min(sample_limit, len(ds)))):
        records.append({
            "task_type": "choice",
            "state": item["text"],
            "question": "What is the customer banking request category?",
            "target_idx": item["label"],
            "target_label": label_names[item["label"]],
            "options": label_names
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[*] Saved {len(records)} Banking77 records to {out_file}")
    return out_file


def download_ag_news(sample_limit: int = 2000) -> str:
    """
    Downloads AG News for topic classification.
    """
    out_file = os.path.join(CACHE_DIR, "ag_news.json")
    if os.path.exists(out_file):
        print(f"[*] AG News already cached at {out_file}")
        return out_file

    print("[*] Downloading AG News from Hugging Face Hub...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    classes = ["World", "Sports", "Business", "Sci/Tech"]

    records = []
    for item in ds.select(range(min(sample_limit, len(ds)))):
        records.append({
            "task_type": "choice",
            "state": item["text"],
            "question": "What is the primary topic of this news report?",
            "target_idx": item["label"],
            "target_label": classes[item["label"]],
            "options": classes
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[*] Saved {len(records)} AG News records to {out_file}")
    return out_file


def download_sst5(sample_limit: int = 2000) -> str:
    """
    Downloads SST-5 (Stanford Sentiment Treebank) for Score evaluation (1-5 scale).
    """
    out_file = os.path.join(CACHE_DIR, "sst5.json")
    if os.path.exists(out_file):
        print(f"[*] SST-5 already cached at {out_file}")
        return out_file

    print("[*] Downloading SST-5 from Hugging Face Hub...")
    try:
        ds = load_dataset("SetFit/sst5", split="train")
        score_mapping = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
        records = []
        for item in ds.select(range(min(sample_limit, len(ds)))):
            records.append({
                "task_type": "score",
                "state": item["text"],
                "question": "Rate the sentiment of this text from 1.0 (very negative) to 5.0 (very positive)",
                "target_score": score_mapping[item["label"]],
                "min_score": 1.0,
                "max_score": 5.0
            })
    except Exception as e:
        print(f"[!] Falling back to synthetic score dataset due to: {e}")
        records = generate_synthetic_score_records(sample_limit)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[*] Saved {len(records)} SST-5 records to {out_file}")
    return out_file


def download_boolq(sample_limit: int = 2000) -> str:
    """
    Downloads BoolQ for Noul binary hypothesis validation.
    """
    out_file = os.path.join(CACHE_DIR, "boolq.json")
    if os.path.exists(out_file):
        print(f"[*] BoolQ already cached at {out_file}")
        return out_file

    print("[*] Downloading BoolQ from Hugging Face Hub...")
    try:
        ds = load_dataset("google/boolq", split="train")
        records = []
        for item in ds.select(range(min(sample_limit, len(ds)))):
            records.append({
                "task_type": "noul",
                "state": item["passage"],
                "hypothesis": item["question"],
                "target_bool": bool(item["answer"])
            })
    except Exception as e:
        print(f"[!] Falling back to synthetic noul dataset due to: {e}")
        records = generate_synthetic_noul_records(sample_limit)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[*] Saved {len(records)} BoolQ records to {out_file}")
    return out_file


def generate_synthetic_score_records(count: int = 500) -> List[Dict[str, Any]]:
    records = []
    samples = [
        ("This is the worst experience of my life, utterly terrible service.", 1.0),
        ("Disappointing quality, broke within two days of usage.", 2.0),
        ("Average product, does the job but nothing extraordinary.", 3.0),
        ("Very good quality, arrived promptly and works well.", 4.0),
        ("Absolutely outstanding, exceeded all my expectations in every way!", 5.0)
    ]
    for i in range(count):
        text, score = samples[i % len(samples)]
        records.append({
            "task_type": "score",
            "state": f"Review #{i}: {text}",
            "question": "Rate the sentiment from 1.0 to 5.0",
            "target_score": float(score),
            "min_score": 1.0,
            "max_score": 5.0
        })
    return records


def generate_synthetic_noul_records(count: int = 500) -> List[Dict[str, Any]]:
    records = []
    samples = [
        ("The server cluster CPU usage is at 98% with recurring memory thrashing.", "The system is under critical resource strain.", True),
        ("The user successfully authenticated with 2-factor OTP.", "The user passed multi-factor authentication.", True),
        ("All unit tests passed with 100% coverage and zero regression errors.", "The build has failed due to syntax errors.", False),
        ("The shipment has arrived at the destination sorting facility.", "The item has already been delivered to the customer door.", False)
    ]
    for i in range(count):
        state, hyp, ans = samples[i % len(samples)]
        records.append({
            "task_type": "noul",
            "state": f"Observation #{i}: {state}",
            "hypothesis": hyp,
            "target_bool": ans
        })
    return records


if __name__ == "__main__":
    download_ag_news(sample_limit=500)
    download_sst5(sample_limit=500)
    download_boolq(sample_limit=500)
