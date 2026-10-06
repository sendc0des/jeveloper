"""
Unit tests for calibration metrics, proper scoring rules, and conformal prediction.
"""
import numpy as np
import torch
import pytest
from calibration.metrics import compute_ece, compute_brier_score, compute_nll
from calibration.proper_scoring import BrierScoreLoss, RLCDCalibrationLoss
from calibration.conformal import ConformalPredictor


def test_ece_perfect_and_worst_calibration():
    # 1. Perfect calibration: confidence equals empirical accuracy
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    targets = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 9/10 correct => accuracy 0.9 in bin ~0.9

    ece_res = compute_ece(confidences, predictions, targets, num_bins=10)
    assert ece_res["ece"] >= 0.0
    assert ece_res["ece"] <= 0.2

    # 2. Brier score calculation
    probs = np.array([[0.9, 0.1], [0.8, 0.2]])
    targs = np.array([0, 0])
    bs = compute_brier_score(probs, targs)
    assert bs > 0.0
    assert bs < 0.1


def test_conformal_prediction_coverage():
    np.random.seed(42)
    n_cal = 500
    n_test = 200
    k_classes = 3

    # Generate synthetic softmax probabilities
    cal_probs = np.random.dirichlet(np.ones(k_classes), size=n_cal)
    cal_targets = np.random.choice(k_classes, size=n_cal)

    test_probs = np.random.dirichlet(np.ones(k_classes), size=n_test)
    test_targets = np.random.choice(k_classes, size=n_test)

    cp = ConformalPredictor(default_alpha=0.10)
    cp.calibrate(cal_probs, cal_targets, alphas=[0.10])

    eval_res = cp.evaluate_coverage(test_probs, test_targets, alpha=0.10)
    assert eval_res["empirical_coverage"] > 0.80  # Target is 90%
    assert eval_res["average_set_size"] >= 1.0


def test_rlcd_loss_gradient():
    rlcd_loss = RLCDCalibrationLoss()
    logits = torch.randn(4, 5, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss_dict = rlcd_loss(logits, probs, targets)
    assert "loss" in loss_dict
    assert "brier_loss" in loss_dict
    assert "calibration_gap" in loss_dict

    loss_dict["loss"].backward()
    assert logits.grad is not None
