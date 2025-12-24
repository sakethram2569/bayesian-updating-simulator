# src/strategies.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from bayes import beta_posterior_update


@dataclass
class BayesianStrategy:
    name: str
    alpha0: float
    beta0: float

    def reset(self):
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.ones = 0
        self.total = 0

    def update(self, s: int):
        if s not in (0, 1):
            raise ValueError("Observation must be 0 or 1")

        self.ones += s
        self.total += 1

        post = beta_posterior_update(
            self.alpha0,
            self.beta0,
            heads=self.ones,
            tails=self.total - self.ones,
        )
        self.alpha, self.beta = post.alpha, post.beta

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / (((a + b) ** 2) * (a + b + 1))


def make_bayesian_strategies():
    """
    Three Bayesian strategies with different priors.
    """
    return [
        BayesianStrategy("Bayesian_weak", 1.0, 1.0),          # uniform
        BayesianStrategy("Bayesian_strong_neutral", 50.0, 50.0),
        BayesianStrategy("Bayesian_strong_wrong", 80.0, 20.0),
    ]


def naive_frequency_estimator(s_obs: np.ndarray) -> np.ndarray:
    """
    Naive estimator:
    p_hat[t] = (#ones up to t) / (t+1)
    """
    cumsum = np.cumsum(s_obs)
    t = np.arange(1, len(s_obs) + 1)
    return cumsum / t
