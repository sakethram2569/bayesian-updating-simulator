# src/metrics.py
from __future__ import annotations
import numpy as np
from bayes import bernoulli_log_loss, clip_prob


def final_abs_error(p_hat: np.ndarray, p_true: float) -> float:
    return float(abs(p_hat[-1] - p_true))


def mean_abs_error(p_hat: np.ndarray, p_true: float) -> float:
    return float(np.mean(np.abs(p_hat - p_true)))


def average_log_loss(p_hat: np.ndarray, x_true: np.ndarray) -> float:
    losses = [
        bernoulli_log_loss(float(clip_prob(p)), int(x))
        for p, x in zip(p_hat, x_true)
    ]
    return float(np.mean(losses))
