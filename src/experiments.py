# src/experiments.py
from __future__ import annotations
import numpy as np

from simulate import simulate_true_bernoulli, apply_label_noise
from strategies import make_bayesian_strategies, naive_frequency_estimator
from metrics import final_abs_error, mean_abs_error


def run_many_paths(
    N: int,
    T: int,
    p_true: float,
    noise: float,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    strategies = make_bayesian_strategies()
    names = [st.name for st in strategies] + ["Naive_freq"]

    final_errs = {name: [] for name in names}
    mean_errs = {name: [] for name in names}

    for _ in range(N):
        x_true = simulate_true_bernoulli(T, p_true, rng)
        s_obs = apply_label_noise(x_true, noise, rng)

        for st in strategies:
            st.reset()

        traj = {st.name: np.zeros(T) for st in strategies}

        for t in range(T):
            s = int(s_obs[t])
            for st in strategies:
                st.update(s)
                traj[st.name][t] = st.mean()

        naive_traj = naive_frequency_estimator(s_obs)

        for st in strategies:
            final_errs[st.name].append(
                final_abs_error(traj[st.name], p_true)
            )
            mean_errs[st.name].append(
                mean_abs_error(traj[st.name], p_true)
            )

        final_errs["Naive_freq"].append(
            final_abs_error(naive_traj, p_true)
        )
        mean_errs["Naive_freq"].append(
            mean_abs_error(naive_traj, p_true)
        )

    summary = {}
    for name in names:
        summary[name] = {
            "final_err": float(np.mean(final_errs[name])),
            "mean_err": float(np.mean(mean_errs[name])),
        }

    return summary


def main():
    N = 1000
    T = 200
    p_true = 0.7
    noises = [0.0, 0.1, 0.2, 0.3]

    for noise in noises:
        summary = run_many_paths(N, T, p_true, noise, seed=42)
        print(f"\nNoise = {noise}")
        for name, stats in summary.items():
            print(
                f"{name:25s} "
                f"final_err={stats['final_err']:.4f} "
                f"mean_err={stats['mean_err']:.4f}"
            )


if __name__ == "__main__":
    main()
