"""
Experiment 2: Robustness to Noise

Analyzes how ICA algorithms handle noisy data:
- FastICA (reference baseline)
- SGD-ICA (stochastic gradient descent)
- Adam-ICA (Adam optimizer)

Compares convergence (objective function) and final solution quality (Amari index)
across different noise levels.
"""

import argparse
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import adam_ica, amari_index, fastica_unmixing_matrix, sgd_ica
from src.utils import center_whiten, generate_synthetic_data


def run_noise_robustness_experiment(
    d=10,
    n=5000,
    noise_levels=(0.0, 0.1, 0.2, 0.5),
    n_iter=2000,
    batch_size=64,
    lr_sgd=0.01,
    lr_adam=0.0005,
    seeds=(0, 1, 2),
):
    """
    Run robustness analysis across noise levels.
    
    For each noise level and seed, compare FastICA, SGD-ICA, and Adam-ICA
    on convergence and final Amari.
    """
    rows = []
    
    for noise_level in noise_levels:
        for seed in seeds:
            print(f"[Noise Robustness] noise={noise_level}, seed={seed} - Generating data...", flush=True)
            
            # Generate synthetic data
            source_types = ["laplace"] * d
            x, s_true, a = generate_synthetic_data(
                D=d,
                N=n,
                source_types=source_types,
                seed=seed,
            )
            
            # Add Gaussian noise
            if noise_level > 0:
                rng = np.random.default_rng(seed)
                noise = rng.standard_normal(x.shape) * noise_level
                x_noisy = x + noise
            else:
                x_noisy = x
            
            xw, mu, w_white = center_whiten(x_noisy)
            
            # --- FastICA ---
            print(f"  FastICA...", flush=True)
            t0 = time.perf_counter()
            v_fica, _, _, _, _ = fastica_unmixing_matrix(
                x_noisy,
                whitening_fn=center_whiten,
                n_components=d,
                max_iter=n_iter,
                tol=1e-8,
                seed=seed,
            )
            t_fica = time.perf_counter() - t0
            amari_fica = amari_index(v_fica @ a)
            
            rows.append({
                "noise_level": noise_level,
                "seed": seed,
                "algorithm": "FastICA",
                "final_amari": amari_fica,
                "time_s": t_fica,
            })
            
            # --- SGD-ICA ---
            print(f"  SGD-ICA...", flush=True)
            t0 = time.perf_counter()
            w_sgd, history_sgd = sgd_ica(
                xw,
                n_iter=n_iter,
                lr=lr_sgd,
                batch_size=batch_size,
                seed=seed,
                fun='logcosh',
            )
            t_sgd = time.perf_counter() - t0
            
            v_sgd = w_sgd @ w_white
            amari_sgd = amari_index(v_sgd @ a)
            
            # Record final objective and Amari
            rows.append({
                "noise_level": noise_level,
                "seed": seed,
                "algorithm": "SGD-ICA",
                "final_objective": history_sgd[-1],
                "final_amari": amari_sgd,
                "time_s": t_sgd,
            })
            
            # --- Adam-ICA ---
            print(f"  Adam-ICA...", flush=True)
            t0 = time.perf_counter()
            w_adam, history_adam = adam_ica(
                xw,
                n_iter=n_iter,
                lr=lr_adam,
                batch_size=batch_size,
                seed=seed,
                fun='logcosh',
            )
            t_adam = time.perf_counter() - t0
            
            v_adam = w_adam @ w_white
            amari_adam = amari_index(v_adam @ a)
            
            rows.append({
                "noise_level": noise_level,
                "seed": seed,
                "algorithm": "Adam-ICA",
                "final_objective": history_adam[-1],
                "final_amari": amari_adam,
                "time_s": t_adam,
            })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Noise Robustness Analysis")
    parser.add_argument("--d", type=int, default=10, help="Dimension (default: 10)")
    parser.add_argument("--n", type=int, default=5000, help="Number of samples (default: 5000)")
    parser.add_argument("--n-iter", type=int, default=2000, help="Max iterations (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr-sgd", type=float, default=0.01, help="SGD learning rate (default: 0.01)")
    parser.add_argument("--lr-adam", type=float, default=0.0005, help="Adam learning rate (default: 0.0005)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("NOISE ROBUSTNESS ANALYSIS EXPERIMENT")
    print("=" * 70)
    print(f"Parameters: D={args.d}, N={args.n}, n_iter={args.n_iter}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Run experiment
    df = run_noise_robustness_experiment(
        d=args.d,
        n=args.n,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        lr_sgd=args.lr_sgd,
        lr_adam=args.lr_adam,
    )
    
    # Save raw results
    csv_path = output_dir / "noise_robustness_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to: {csv_path}")
    
    # Compute summary statistics
    summary = df.groupby(["noise_level", "algorithm"]).agg({
        "final_amari": ["mean", "std"],
        "time_s": "mean",
    }).round(6)
    
    summary_path = output_dir / "noise_robustness_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved to: {summary_path}")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Amari vs Noise Level
    ax = axes[0]
    noise_levels = sorted(df["noise_level"].unique())
    for algo in ["FastICA", "SGD-ICA", "Adam-ICA"]:
        subset = df[df["algorithm"] == algo]
        mean_amari = [subset[subset["noise_level"] == nl]["final_amari"].mean() for nl in noise_levels]
        ax.plot(noise_levels, mean_amari, marker='o', label=algo, markersize=7, linewidth=2.5)
    
    ax.set_xlabel("Noise Level (σ)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Amari Index (lower is better)", fontsize=13, fontweight='bold')
    ax.set_title("Robustness to Noise: Amari vs Noise Level", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Computation Time vs Noise Level
    ax = axes[1]
    for algo in ["FastICA", "SGD-ICA", "Adam-ICA"]:
        subset = df[df["algorithm"] == algo]
        mean_time = [subset[subset["noise_level"] == nl]["time_s"].mean() for nl in noise_levels]
        ax.plot(noise_levels, mean_time, marker='s', label=algo, markersize=7, linewidth=2.5)
    
    ax.set_xlabel("Noise Level (σ)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Time (seconds)", fontsize=13, fontweight='bold')
    ax.set_title("Computation Time vs Noise Level", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    fig_path = output_dir / "noise_robustness_plot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {fig_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("NOISE ROBUSTNESS ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
