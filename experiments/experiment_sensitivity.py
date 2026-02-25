"""
Experiment 2 — Sensitivity to number of samples and SNR.

This experiment evaluates FastICA, Infomax and JADE under varying
conditions:
  1. Varying number of samples (T): measures how performance improves with
     more data.
  2. Varying noise level (SNR): measures robustness to additive white noise.

Results are saved as CSV tables and figures in ``experiments/results/``.

Usage
-----
    python experiments/experiment_sensitivity.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ica import FastICA, InfomaxICA, JADE
from ica.utils import generate_sources, mix_sources, amari_error


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_SOURCES = 3
SEED_BASE  = 0
N_TRIALS   = 5        # independent trials per condition
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

SAMPLE_SIZES = [100, 250, 500, 1000, 2000, 5000]
SNR_DB_LIST  = [40, 30, 20, 15, 10, 5]   # dB


def _build_algorithms(seed):
    return {
        "FastICA (deflation)": FastICA(algorithm="deflation", g="logcosh", random_state=seed),
        "FastICA (symmetric)": FastICA(algorithm="symmetric", g="logcosh", random_state=seed),
        "Infomax":             InfomaxICA(extended=False, random_state=seed),
        "Extended Infomax":    InfomaxICA(extended=True,  random_state=seed),
        "JADE":                JADE(),
    }


# ---------------------------------------------------------------------------
# Sub-experiment 1: vary T
# ---------------------------------------------------------------------------

def experiment_sample_size():
    print("=" * 60)
    print("Sub-experiment 1: varying number of samples")
    print("=" * 60)

    algo_names = list(_build_algorithms(0).keys())
    errors = {name: [] for name in algo_names}  # list of mean Amari over trials

    for T in SAMPLE_SIZES:
        trial_errors = {name: [] for name in algo_names}
        for trial in range(N_TRIALS):
            seed = SEED_BASE + trial
            S_true = generate_sources(N_SOURCES, T, seed=seed)
            X, A_true = mix_sources(S_true, seed=seed + 100)
            for name, model in _build_algorithms(seed).items():
                try:
                    model.fit(X)
                    err = amari_error(model.components_, A_true)
                except Exception:
                    err = np.nan
                trial_errors[name].append(err)

        for name in algo_names:
            mean_err = np.nanmean(trial_errors[name])
            errors[name].append(mean_err)
            print(f"  T={T:5d}  {name:25s}  Amari={mean_err:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in algo_names:
        ax.plot(SAMPLE_SIZES, errors[name], marker="o", label=name)
    ax.set_xlabel("Number of samples T")
    ax.set_ylabel("Mean Amari error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Algorithm performance vs. number of samples")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sensitivity_samples.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Figure saved → {path}\n")
    return errors


# ---------------------------------------------------------------------------
# Sub-experiment 2: vary SNR
# ---------------------------------------------------------------------------

def _add_noise(X: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """Add white Gaussian noise to achieve the given SNR (in dB)."""
    signal_power = (X ** 2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.standard_normal(X.shape) * np.sqrt(noise_power)
    return X + noise


def experiment_snr():
    print("=" * 60)
    print("Sub-experiment 2: varying SNR")
    print("=" * 60)

    T = 2000
    algo_names = list(_build_algorithms(0).keys())
    errors = {name: [] for name in algo_names}

    for snr_db in SNR_DB_LIST:
        trial_errors = {name: [] for name in algo_names}
        for trial in range(N_TRIALS):
            seed = SEED_BASE + trial
            rng = np.random.default_rng(seed)
            S_true = generate_sources(N_SOURCES, T, seed=seed)
            X, A_true = mix_sources(S_true, seed=seed + 100)
            Xn = _add_noise(X, snr_db, rng)

            for name, model in _build_algorithms(seed).items():
                try:
                    model.fit(Xn)
                    err = amari_error(model.components_, A_true)
                except Exception:
                    err = np.nan
                trial_errors[name].append(err)

        for name in algo_names:
            mean_err = np.nanmean(trial_errors[name])
            errors[name].append(mean_err)
            print(f"  SNR={snr_db:3d} dB  {name:25s}  Amari={mean_err:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in algo_names:
        ax.plot(SNR_DB_LIST, errors[name], marker="o", label=name)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean Amari error")
    ax.invert_xaxis()   # high SNR on the left
    ax.set_title("Algorithm performance vs. noise level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sensitivity_snr.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Figure saved → {path}\n")
    return errors


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    experiment_sample_size()
    experiment_snr()
