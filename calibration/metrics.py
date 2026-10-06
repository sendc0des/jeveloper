"""
Calibration metrics and statistical scoring rules for decision models.
Includes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Brier Score, and reliability diagram generation.
"""
import numpy as np
from typing import Dict, List, Tuple, Union


def compute_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes multi-class Brier score.
    probabilities: (N, K) array of predicted probabilities
    targets: (N,) array of true class integer indices (0 to K-1)
    """
    N, K = probabilities.shape
    one_hot = np.zeros((N, K), dtype=np.float32)
    one_hot[np.arange(N), targets] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def compute_binary_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes binary Brier score.
    probabilities: (N,) probability of positive class
    targets: (N,) binary true labels {0, 1}
    """
    return float(np.mean((probabilities - targets) ** 2))


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10
) -> Dict[str, Union[float, Dict]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        confidences: (N,) float array of predicted confidence in [0, 1]
        predictions: (N,) int array of predicted classes
        targets: (N,) int array of ground truth classes
        num_bins: Number of confidence bins (standard is 10 or 15)
        
    Returns:
        Dict with 'ece', 'mce', and 'bins_data' for plotting reliability diagrams.
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    accuracies = (predictions == targets).astype(float)
    
    ece = 0.0
    mce = 0.0
    N = len(confidences)
    
    bins_data = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # In bin mask
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_gap = abs(bin_acc - bin_conf)
            
            ece += (bin_size / N) * bin_gap
            mce = max(mce, bin_gap)
            
            bins_data.append({
                "bin_idx": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_center": float((bin_lower + bin_upper) / 2.0),
                "count": int(bin_size),
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": bin_gap
            })
        else:
            bins_data.append({
                "bin_idx": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_center": float((bin_lower + bin_upper) / 2.0),
                "count": 0,
                "accuracy": 0.0,
                "confidence": float((bin_lower + bin_upper) / 2.0),
                "gap": 0.0
            })
            
    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins_data": bins_data
    }


def compute_nll(probabilities: np.ndarray, targets: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes Negative Log-Likelihood (log loss).
    """
    N = len(targets)
    clipped = np.clip(probabilities, eps, 1.0)
    picked = clipped[np.arange(N), targets]
    return float(-np.mean(np.log(picked)))
