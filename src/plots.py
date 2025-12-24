# src/plots.py
import os
import numpy as np
import matplotlib.pyplot as plt

from simulate import simulate_true_bernoulli, apply_label_noise
from strategies import make_bayesian_strategies, naive_frequency_estimator


def plot_convergence(T=200, p_true=0.7, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    x_true = simulate_true_bernoulli(T, p_true, rng)
    s_obs = apply_label_noise(x_true, noise, rng)

    strategies = make_bayesian_strategies()
    for st in strategies:
        st.reset()

    traj = {st.name: np.zeros(T) for st in strategies}

    for t in range(T):
        s = int(s_obs[t])
        for st in strategies:
            st.update(s)
            traj[st.name][t] = st.mean()

    naive = naive_frequency_estimator(s_obs)

    os.makedirs("results/figures", exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, arr in traj.items():
        plt.plot(arr, label=name)
    plt.plot(naive, label="Naive_freq", linestyle=":")
    plt.axhline(p_true, color="black", linestyle="--", label="True p")
    plt.xlabel("Time")
    plt.ylabel("Estimate of p")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/convergence.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_convergence()
