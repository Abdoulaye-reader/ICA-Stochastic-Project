"""
Hyperparameter optimization for SGD-ICA and Adam-ICA.

This script performs grid-search on (lr, n_iter, batch_size) and reports
robust averages across several seeds.
"""

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import adam_ica, amari_index, sgd_ica
from src.utils import center_whiten, generate_synthetic_data


def evaluate_config(algo, lr, n_iter, batch_size, d, n, seeds):
    rows = []

    for sd in seeds:
        source_types = ["laplace"] * d
        x, _, a = generate_synthetic_data(D=d, N=n, source_types=source_types, seed=sd)
        xw, _, w_white = center_whiten(x)

        t0 = time.perf_counter()
        if algo == "SGD-ICA":
            w, _ = sgd_ica(xw, n_iter=n_iter, lr=lr, batch_size=batch_size, seed=sd)
        elif algo == "Adam-ICA":
            w, _ = adam_ica(xw, n_iter=n_iter, lr=lr, batch_size=batch_size, seed=sd)
        else:
            raise ValueError(f"Unknown algo: {algo}")
        t_elapsed = time.perf_counter() - t0

        c = (w @ w_white) @ a
        amari = amari_index(c)

        rows.append(
            {
                "algo": algo,
                "D": d,
                "N": n,
                "seed": sd,
                "lr": lr,
                "n_iter": n_iter,
                "batch_size": batch_size,
                "amari": amari,
                "time_s": t_elapsed,
            }
        )

    return rows


def run_grid(algo, d, n, seeds, lrs, n_iters, batch_sizes):
    rows = []
    total = len(lrs) * len(n_iters) * len(batch_sizes)
    k = 0

    for lr in lrs:
        for n_iter in n_iters:
            for batch_size in batch_sizes:
                k += 1
                print(
                    f"[{algo}] config {k}/{total}: lr={lr}, n_iter={n_iter}, batch_size={batch_size}",
                    flush=True,
                )
                rows.extend(
                    evaluate_config(
                        algo=algo,
                        lr=lr,
                        n_iter=n_iter,
                        batch_size=batch_size,
                        d=d,
                        n=n,
                        seeds=seeds,
                    )
                )

    return pd.DataFrame(rows)


def summarize(df):
    summary = (
        df.groupby(["algo", "lr", "n_iter", "batch_size"])
        .agg(
            amari_mean=("amari", "mean"),
            amari_std=("amari", "std"),
            time_mean=("time_s", "mean"),
            time_std=("time_s", "std"),
        )
        .reset_index()
    )

    # Main criterion: Amari. Secondary criterion: runtime.
    summary = summary.sort_values(["algo", "amari_mean", "time_mean"], ascending=[True, True, True])
    return summary


def print_top(summary, top_k):
    for algo in ["SGD-ICA", "Adam-ICA"]:
        print(f"\n=== Top {top_k} for {algo} ===")
        sub = summary[summary["algo"] == algo].head(top_k)
        for _, r in sub.iterrows():
            print(
                f"lr={r['lr']:.6g}, n_iter={int(r['n_iter'])}, batch_size={int(r['batch_size'])} "
                f"| Amari={r['amari_mean']:.4f} +/- {r['amari_std']:.4f} "
                f"| Time={r['time_mean']:.3f}s +/- {r['time_std']:.3f}s"
            )


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_sgd = run_grid(
        algo="SGD-ICA",
        d=args.d,
        n=args.n,
        seeds=args.seeds,
        lrs=args.sgd_lrs,
        n_iters=args.n_iters,
        batch_sizes=args.batch_sizes,
    )

    df_adam = run_grid(
        algo="Adam-ICA",
        d=args.d,
        n=args.n,
        seeds=args.seeds,
        lrs=args.adam_lrs,
        n_iters=args.n_iters,
        batch_sizes=args.batch_sizes,
    )

    df_all = pd.concat([df_sgd, df_adam], ignore_index=True)
    summary = summarize(df_all)

    raw_path = output_dir / "hyperparameter_search_raw.csv"
    summary_path = output_dir / "hyperparameter_search_summary.csv"
    df_all.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print_top(summary, top_k=args.top_k)
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search hyperparameters for SGD-ICA and Adam-ICA")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")

    parser.add_argument("--d", type=int, default=10, help="Dimension D")
    parser.add_argument("--n", type=int, default=5000, help="Number of samples N")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds")

    parser.add_argument("--sgd_lrs", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02], help="Learning rates for SGD-ICA")
    parser.add_argument("--adam_lrs", type=float, nargs="+", default=[0.0001, 0.0005, 0.001, 0.002], help="Learning rates for Adam-ICA")
    parser.add_argument("--n_iters", type=int, nargs="+", default=[300, 500, 1000], help="Iteration counts")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 64, 128], help="Batch sizes")
    parser.add_argument("--top_k", type=int, default=5, help="How many best configs to print")

    main(parser.parse_args())
