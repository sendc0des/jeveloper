"""
Example 2: Continuous Urgency & Incident Severity Scorer.
Demonstrates bounded continuous Score evaluation with expected value and distributional variance.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.jev import JevClient


def run_urgency_scorer():
    print("=== JEVLOPER: Incident Severity Scorer ===")
    client = JevClient()

    incident_log = (
        "CRITICAL ALERT: Production database primary replica unresponsive for 120 seconds. "
        "Failover initiated. 42 downstream microservices reporting connection pool starvation. "
        "Customer-facing checkout gateway returning 503 errors."
    )

    print(f"\n[Incident State]:\n\"{incident_log}\"")
    print("\n[Evaluating Score...]")

    res = client.score(
        state=incident_log,
        question="Rate the operational severity level from 1.0 (trivial) to 5.0 (catastrophic)",
        min_score=1.0,
        max_score=5.0
    )

    print("\n--- Severity Result ---")
    print(f"Calculated Score: {res.score:.2f} / 5.0")
    print(f"Uncertainty (Var): {res.variance:.4f}")
    print(f"Inference Time  : {res.latency_ms:.2f} ms")
    print("\nProbability Mass across Bins:")
    for bin_center, prob in res.distribution.items():
        bar = "█" * int(prob * 30)
        print(f"  Level {bin_center:<4.1f} : {prob * 100:>5.1f}% | {bar}")


if __name__ == "__main__":
    run_urgency_scorer()
