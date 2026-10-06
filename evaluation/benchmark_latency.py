"""
Hardware Latency and Throughput Profiler.
Measures p50, p95, and p99 inference latency and QPS on RTX 4050 / CUDA.
"""
import time
import torch
import numpy as np
from typing import Dict, Any, List
from client.jev import JevClient


def benchmark_client_latency(
    client: JevClient,
    iterations: int = 50,
    warmup: int = 10
) -> Dict[str, Any]:
    """
    Benchmarks end-to-end latency for Choice, Score, and Noul primitives.
    """
    device = client.device
    print(f"[*] Running latency benchmark on {device} ({iterations} iterations)...")

    # Sample tasks
    sample_choice = {
        "state": "Customer: My card was declined at an ATM in Chicago while I am abroad.",
        "question": "What is the primary customer intent?",
        "options": ["card_declined", "foreign_transaction", "fraud_alert", "pin_reset", "balance_query"]
    }
    sample_score = {
        "state": "The user reported system-wide API outage affecting 80% of customer traffic.",
        "question": "Rate incident severity from 1.0 to 5.0"
    }
    sample_noul = {
        "state": "User authenticated with verified biometric security token.",
        "hypothesis": "The session has verified biometric authentication."
    }

    # Warmup
    for _ in range(warmup):
        client.choice(**sample_choice)
        client.score(**sample_score)
        client.noul(**sample_noul)

    if device == "cuda":
        torch.cuda.synchronize()

    # Choice Benchmarks
    choice_latencies = []
    for _ in range(iterations):
        res = client.choice(**sample_choice)
        choice_latencies.append(res.latency_ms)

    # Score Benchmarks
    score_latencies = []
    for _ in range(iterations):
        res = client.score(**sample_score)
        score_latencies.append(res.latency_ms)

    # Noul Benchmarks
    noul_latencies = []
    for _ in range(iterations):
        res = client.noul(**sample_noul)
        noul_latencies.append(res.latency_ms)

    def stats(arr: List[float]) -> Dict[str, float]:
        np_arr = np.array(arr)
        return {
            "p50_ms": float(np.percentile(np_arr, 50)),
            "p95_ms": float(np.percentile(np_arr, 95)),
            "p99_ms": float(np.percentile(np_arr, 99)),
            "mean_ms": float(np.mean(np_arr)),
            "throughput_qps": float(1000.0 / np.mean(np_arr))
        }

    results = {
        "choice": stats(choice_latencies),
        "score": stats(score_latencies),
        "noul": stats(noul_latencies)
    }

    print("\n--- Latency Benchmark Summary ---")
    for task, s in results.items():
        print(f"[{task.upper()}] p50: {s['p50_ms']:.2f}ms | p95: {s['p95_ms']:.2f}ms | Throughput: {s['throughput_qps']:.1f} req/s")

    return results


if __name__ == "__main__":
    c = JevClient()
    benchmark_client_latency(c, iterations=30)
