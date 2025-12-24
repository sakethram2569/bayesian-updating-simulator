# Bayesian Updating Simulator under Noisy Signals

## Overview
This project studies how different belief-update strategies perform when estimating an unknown probability from noisy observations. The setting models a trader receiving imperfect signals about an underlying state.

## Problem Setup
- Hidden variable: unknown Bernoulli probability p
- Observations are corrupted by label noise
- Goal: estimate p over time under uncertainty

## Methods
- Bayesian updating with Beta–Bernoulli conjugacy
- Multiple priors: weak, strong neutral, strong misspecified
- Naive frequency estimator as baseline
- Monte Carlo evaluation across noise regimes

## Metrics
- Final absolute error
- Mean absolute error over time

## Key Findings
- Weak Bayesian and naive estimators adapt quickly but suffer under noise
- Strong priors reduce variance and outperform in high-noise regimes
- No single strategy dominates; bias–variance tradeoff is central

### Identifiability Note
Because observations are corrupted by unknown label noise, the true Bernoulli parameter is not identifiable from the data. Estimators converge to an effective probability induced by the noise process rather than the latent truth. This explains why no strategy necessarily converges to the true p even with many observations.

## How to Run
```bash
pip install -r requirements.txt
python src/experiments.py
python src/plots.py
