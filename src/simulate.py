# src/simulate.py
from __future__ import annotations
import numpy as np


def simulate_true_bernoulli(T: int, p_true: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate true hidden outcomes x_t ~ Bernoulli(p_true)
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    if not (0.0 <= p_true <= 1.0):
        raise ValueError("p_true must be in [0,1]")

    x = rng.binomial(n=1, p=p_true, size=T).astype(int)
    return x


def apply_label_noise(x: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """
    Flip each bit with probability = noise.
    noise = 0.0  -> perfect observation
    noise = 0.5  -> almost useless signal
    """
    if not (0.0 <= noise <= 1.0):
        raise ValueError("noise must be in [0,1]")

    flips = rng.binomial(n=1, p=noise, size=len(x)).astype(int)
    s = x ^ flips
    return s.astype(int)
