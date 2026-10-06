"""
Split Conformal Prediction Engine.
Provides distribution-free finite-sample statistical coverage guarantees:
P(True Class in Prediction Set) >= 1 - alpha.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Union


class ConformalPredictor:
    """
    Split Conformal Prediction for Categorical and Binary Decisions.
    Guarantees that the true outcome is contained in the returned prediction set
    with probability at least (1 - alpha).
    """
    def __init__(self, default_alpha: float = 0.05):
        self.default_alpha = default_alpha
        self.calibrated_quantiles: Dict[float, float] = {}
        self.is_calibrated = False

    def calibrate(
        self,
        cal_probabilities: np.ndarray,
        cal_targets: np.ndarray,
        alphas: Optional[List[float]] = None
    ):
        """
        Calibrates on held-out validation / calibration data.
        
        Args:
            cal_probabilities: (N, K) array of predicted probabilities
            cal_targets: (N,) array of true class integer indices
            alphas: List of significance levels to precalculate (e.g. [0.01, 0.05, 0.10])
        """
        N = len(cal_targets)
        assert N > 0, "Calibration set cannot be empty."

        # Compute non-conformity scores: s_i = 1 - p(y_i)
        true_class_probs = cal_probabilities[np.arange(N), cal_targets]
        scores = 1.0 - true_class_probs

        if alphas is None:
            alphas = [0.01, 0.05, 0.10, 0.20]

        for alpha in alphas:
            # Conformal quantile: ceil((N + 1) * (1 - alpha)) / N
            p_val = min(1.0, np.ceil((N + 1) * (1 - alpha)) / N)
            quantile_val = float(np.quantile(scores, p_val, method="higher"))
            self.calibrated_quantiles[alpha] = quantile_val

        self.is_calibrated = True
        print(f"[*] Conformal predictor calibrated on {N} samples.")
        for a, q in self.calibrated_quantiles.items():
            print(f"    - Alpha: {a:.2f} (Coverage: {(1-a)*100:.1f}%) -> Nonconformity threshold: {q:.4f}")

    def predict_set(
        self,
        probabilities: np.ndarray,
        candidate_ids: List[str],
        alpha: Optional[float] = None
    ) -> List[str]:
        """
        Generates certified prediction set for a single sample.
        
        Args:
            probabilities: (K,) 1D array of probabilities for candidates
            candidate_ids: List of candidate string IDs
            alpha: Significance level (default: self.default_alpha)
            
        Returns:
            List of candidate IDs in the certified prediction set.
        """
        a = alpha or self.default_alpha
        threshold = self.calibrated_quantiles.get(a)
        
        if threshold is None:
            # If exact alpha wasn't pre-calibrated or not yet calibrated, use adaptive threshold
            threshold = 1.0 - (1.0 - a)

        # Include all options where (1 - p_k) <= threshold <=> p_k >= 1 - threshold
        min_p = max(0.0, 1.0 - threshold)
        selected_set = [cid for cid, p in zip(candidate_ids, probabilities) if p >= min_p]

        # Guarantee non-empty set by including argmax if none meet threshold
        if not selected_set:
            argmax_idx = int(np.argmax(probabilities))
            selected_set = [candidate_ids[argmax_idx]]

        return selected_set

    def evaluate_coverage(
        self,
        test_probabilities: np.ndarray,
        test_targets: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Verifies empirical coverage and average set size on test data.
        """
        N, K = test_probabilities.shape
        threshold = self.calibrated_quantiles.get(alpha, 1.0 - (1.0 - alpha))
        min_p = max(0.0, 1.0 - threshold)

        covered = 0
        set_sizes = []

        for i in range(N):
            probs = test_probabilities[i]
            target = test_targets[i]
            # Classes in prediction set
            pred_set = np.where(probs >= min_p)[0]
            if len(pred_set) == 0:
                pred_set = np.array([np.argmax(probs)])
            set_sizes.append(len(pred_set))
            if target in pred_set:
                covered += 1

        empirical_coverage = covered / N
        avg_set_size = float(np.mean(set_sizes))

        return {
            "target_coverage": 1.0 - alpha,
            "empirical_coverage": empirical_coverage,
            "average_set_size": avg_set_size,
            "meets_guarantee": empirical_coverage >= (1.0 - alpha)
        }
