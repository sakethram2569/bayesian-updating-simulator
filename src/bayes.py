# src/bayes.py
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BetaPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        denom = (a + b) ** 2 * (a + b + 1.0)
        return (a * b) / denom


def beta_posterior_update(alpha: float, beta: float, heads: int, tails: int) -> BetaPosterior:
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be > 0")
    if heads < 0 or tails < 0:
        raise ValueError("heads and tails must be >= 0")
    return BetaPosterior(alpha + heads, beta + tails)


def clip_prob(p: float, eps: float = 1e-12) -> float:
    return max(eps, min(1.0 - eps, p))


def bernoulli_log_loss(p_hat: float, x: int) -> float:
    if x not in (0, 1):
        raise ValueError("x must be 0 or 1")
    p = clip_prob(p_hat)
    return -(x * math.log(p) + (1 - x) * math.log(1 - p))
