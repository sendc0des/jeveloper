"""
Reliability Diagram and Calibration Curve Generator.
Generates publication-quality calibration plots for research and portfolio reports.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List


def plot_reliability_diagram(
    bins_data: List[Dict[str, Any]],
    ece: float,
    mce: float,
    title: str = "Reliability Diagram (Calibration Curve)",
    save_path: str = "reports/reliability_diagram.png"
):
    """
    Renders and saves a reliability diagram showing confidence vs empirical accuracy.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    bin_centers = [b["bin_center"] for b in bins_data]
    accuracies = [b["accuracy"] for b in bins_data]
    confidences = [b["confidence"] for b in bins_data]
    counts = [b["count"] for b in bins_data]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Top plot: Accuracy vs Confidence
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7)
    
    # Draw bars for empirical accuracy
    bar_width = 1.0 / len(bins_data)
    ax1.bar(
        bin_centers,
        accuracies,
        width=bar_width * 0.85,
        alpha=0.65,
        color="#2563eb",
        edgecolor="#1d4ed8",
        label="Outputs"
    )

    # Shaded calibration gap
    for c, acc, conf in zip(bin_centers, accuracies, confidences):
        if acc != 0 or conf != 0:
            ax1.vlines(c, ymin=min(acc, conf), ymax=max(acc, conf), color="#dc2626", linewidth=2.5, alpha=0.8)

    ax1.set_ylabel("Empirical Accuracy", fontsize=12)
    ax1.set_title(f"{title}\nECE: {ece * 100:.2f}% | MCE: {mce * 100:.2f}%", fontsize=13, fontweight="bold")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Bottom plot: Frequency count per bin
    ax2.bar(bin_centers, counts, width=bar_width * 0.85, color="#64748b", alpha=0.7)
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Saved publication reliability diagram to {save_path}")
