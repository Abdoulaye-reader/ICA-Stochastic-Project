"""
Experiment 1 — Blind Source Separation (BSS) benchmark.

This script generates synthetic independent sources, mixes them, and then
applies FastICA, Infomax and JADE to recover the originals.  It reports
the Amari error and per-source SIR for each algorithm and saves a figure
comparing the original signals with the recovered ones.

Usage
-----
    python experiments/experiment_bss.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from ica import FastICA, InfomaxICA, JADE
from ica.utils import generate_sources, mix_sources, amari_error, signal_to_interference_ratio


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_SOURCES = 4
N_SAMPLES = 2000
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_bss_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Generate and mix sources ---
    S_true = generate_sources(N_SOURCES, N_SAMPLES, seed=SEED)
    X, A_true = mix_sources(S_true, seed=SEED + 1)

    algorithms = {
        "FastICA (deflation, logcosh)": FastICA(
            algorithm="deflation", g="logcosh", random_state=SEED),
        "FastICA (symmetric, exp)":    FastICA(
            algorithm="symmetric", g="exp", random_state=SEED),
        "Infomax":                     InfomaxICA(
            extended=False, random_state=SEED),
        "Extended Infomax":            InfomaxICA(
            extended=True, random_state=SEED),
        "JADE":                        JADE(),
    }

    results = {}
    for name, model in algorithms.items():
        S_est = model.fit_transform(X)
        err = amari_error(model.components_, A_true)
        sir = signal_to_interference_ratio(S_est, S_true)
        results[name] = {
            "S_est":  S_est,
            "amari":  err,
            "sir":    sir,
            "n_iter": model.n_iter_,
        }
        print(f"{name:40s}  Amari={err:.4f}  mean SIR={sir.mean():.1f} dB  "
              f"iters={model.n_iter_}")

    # --- Plot ---
    _plot_signals(S_true, X, results)
    _plot_metrics(results)

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_signals(S_true, X, results):
    algo_names = list(results.keys())
    n_sources = S_true.shape[0]
    n_cols = 2 + len(algo_names)  # True | Mixed | algo1 | algo2 | ...
    t = np.arange(S_true.shape[1])

    fig, axes = plt.subplots(
        n_sources, n_cols,
        figsize=(4 * n_cols, 1.8 * n_sources),
        sharex=True,
    )

    col_titles = ["True sources", "Mixed signals"] + algo_names
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8, pad=4)

    for row in range(n_sources):
        # True
        axes[row, 0].plot(t, S_true[row], lw=0.5, color="steelblue")
        axes[row, 0].set_ylabel(f"s{row + 1}", fontsize=7)
        axes[row, 0].tick_params(labelsize=6)

        # Mixed
        axes[row, 1].plot(t, X[row], lw=0.5, color="orange")
        axes[row, 1].tick_params(labelsize=6)

        # Estimated — find best-matching component
        for col, name in enumerate(algo_names):
            S_est = results[name]["S_est"]
            # Match by maximum absolute correlation with true source
            corr = np.abs(S_est @ S_true[row] / (
                np.linalg.norm(S_est, axis=1) * np.linalg.norm(S_true[row]) + 1e-12
            ))
            best = np.argmax(corr)
            s = S_est[best].copy()
            if np.dot(s, S_true[row]) < 0:
                s = -s
            axes[row, col + 2].plot(t, s, lw=0.5, color="green")
            axes[row, col + 2].tick_params(labelsize=6)

    fig.suptitle("BSS — signal comparison", y=1.01, fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bss_signals.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {path}")


def _plot_metrics(results):
    names = list(results.keys())
    amari_vals = [results[n]["amari"] for n in names]
    sir_vals   = [results[n]["sir"].mean() for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(names))
    width = 0.6

    bars1 = ax1.bar(x, amari_vals, width, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Amari error (lower is better)")
    ax1.set_title("Amari Performance Index")
    for bar, val in zip(bars1, amari_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    bars2 = ax2.bar(x, sir_vals, width, color="seagreen")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Mean SIR (dB) (higher is better)")
    ax2.set_title("Signal-to-Interference Ratio")
    for bar, val in zip(bars2, sir_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("BSS — algorithm comparison", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bss_metrics.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {path}")


if __name__ == "__main__":
    run_bss_experiment()
