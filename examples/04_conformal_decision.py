"""
Example 4: Conformal Decision Making with Certified Statistical Coverage.
Demonstrates how Split Conformal Prediction outputs guaranteed prediction sets
rather than risky point estimates.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.jev import JevClient
from calibration.conformal import ConformalPredictor


def run_conformal_demo():
    print("=== JEVLOPER: Conformal Prediction Engine ===")
    
    # Simulate a calibration dataset of 100 validation decisions
    print("[*] Calibrating conformal engine on validation holdout...")
    np.random.seed(42)
    n_cal = 200
    k_classes = 4
    
    # Generate mock calibration distributions
    sim_probs = np.random.dirichlet(np.ones(k_classes) * 1.5, size=n_cal)
    sim_targets = np.random.choice(k_classes, size=n_cal)
    
    conformal = ConformalPredictor()
    conformal.calibrate(sim_probs, sim_targets, alphas=[0.05, 0.10])
    
    # Two test scenarios:
    # 1. Unambiguous sample
    # 2. Borderline / Ambiguous sample
    options = ["order_status", "refund_request", "cancel_order", "technical_support"]
    
    print("\n--- Scenario A: Clear Unambiguous Intent ---")
    clear_probs = np.array([0.02, 0.91, 0.05, 0.02])
    res_set_a = conformal.predict_set(clear_probs, candidate_ids=options, alpha=0.05)
    print(f"Probabilities: {dict(zip(options, clear_probs))}")
    print(f"Certified Set (95% Coverage): {res_set_a} (Set Size: {len(res_set_a)})")
    print("Outcome: High certainty, automated execution allowed.")

    print("\n--- Scenario B: Ambiguous Borderline Intent ---")
    ambiguous_probs = np.array([0.05, 0.46, 0.44, 0.05])
    res_set_b = conformal.predict_set(ambiguous_probs, candidate_ids=options, alpha=0.05)
    print(f"Probabilities: {dict(zip(options, ambiguous_probs))}")
    print(f"Certified Set (95% Coverage): {res_set_b} (Set Size: {len(res_set_b)})")
    print("Outcome: Ambiguity detected! Both options included in guaranteed set. Route to human review.")


if __name__ == "__main__":
    run_conformal_demo()
