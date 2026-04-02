"""
Convergence Analysis: Stochastic Algorithms

Analyzes convergence of stochastic ICA algorithms over iterations:
- SGD-ICA (stochastic gradient descent)
- Adam-ICA (Adam optimizer)

Shows convergence through the objective function (Infomax proxy) across iterations.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import adam_ica, sgd_ica
from src.utils import center_whiten, generate_synthetic_data


def run_convergence_experiment(
    d=10,
    n=5000,
    n_iter=2000,
    batch_size=64,
    lr_sgd=0.01,
    lr_adam=0.0005,
    seeds=(0, 1, 2),
):
    """
    Run convergence analysis on stochastic algorithms.
    
    For each seed, track the objective function over iterations.
    """
    rows = []
    
    for seed in seeds:
        print(f"[Convergence] D={d}, N={n}, seed={seed} - Generating data...", flush=True)
        
        # Generate synthetic data
        source_types = ["laplace"] * d
        x, s_true, a = generate_synthetic_data(
            D=d,
            N=n,
            source_types=source_types,
            seed=seed,
        )
        xw, mu, w_white = center_whiten(x)
        
        # --- SGD-ICA (track objective at each iteration) ---
        print(f"  Running SGD-ICA...", flush=True)
        w_sgd, history_sgd = sgd_ica(
            xw,
            n_iter=n_iter,
            lr=lr_sgd,
            batch_size=batch_size,
            seed=seed,
            fun='logcosh',
        )
        
        # Record all iterations
        for it, obj_val in enumerate(history_sgd):
            rows.append({
                "seed": seed,
                "algorithm": "SGD-ICA",
                "iteration": it + 1,
                "objective": obj_val,
            })
        
        # --- Adam-ICA (track objective at each iteration) ---
        print(f"  Running Adam-ICA...", flush=True)
        w_adam, history_adam = adam_ica(
            xw,
            n_iter=n_iter,
            lr=lr_adam,
            batch_size=batch_size,
            seed=seed,
            fun='logcosh',
        )
        
        for it, obj_val in enumerate(history_adam):
            rows.append({
                "seed": seed,
                "algorithm": "Adam-ICA",
                "iteration": it + 1,
                "objective": obj_val,
            })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Convergence Analysis")
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
    print("STOCHASTIC ALGORITHMS CONVERGENCE ANALYSIS")
    print("=" * 70)
    print(f"Parameters: D={args.d}, N={args.n}, n_iter={args.n_iter}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Run experiment
    df = run_convergence_experiment(
        d=args.d,
        n=args.n,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        lr_sgd=args.lr_sgd,
        lr_adam=args.lr_adam,
    )
    
    # Save raw results
    csv_path = output_dir / "convergence_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to: {csv_path}")
    
    # Compute summary statistics
    summary = df.groupby("algorithm").agg({
        "objective": ["min", "max", "mean", "std"],
        "iteration": "max",
    }).round(6)
    
    summary_path = output_dir / "convergence_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved to: {summary_path}")
    
    # Create plots
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Objective vs Iteration
    for algo in ["SGD-ICA", "Adam-ICA"]:
        subset = df[df["algorithm"] == algo]
        mean_obj = subset.groupby("iteration")["objective"].mean()
        ax.plot(mean_obj.index, mean_obj.values, marker='o', label=algo, markersize=5, linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel("Iteration", fontsize=14, fontweight='bold')
    ax.set_ylabel("Objective (Infomax proxy)", fontsize=14, fontweight='bold')
    ax.set_title("Stochastic ICA Convergence", fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    fig_path = output_dir / "convergence_plot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {fig_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("STOCHASTIC CONVERGENCE ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
