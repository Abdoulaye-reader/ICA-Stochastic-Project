"""
Experiment 1: Scalability analysis mirroring notebook experiments.

Two complementary studies are run:
1) Quality/time vs dimension D, with fixed N.
2) Quality/time vs number of samples N, with fixed D and fixed epochs for stochastic methods.
"""

import argparse
import math
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import adam_ica, amari_index, fastica_unmixing_matrix, sgd_ica
from src.utils import center_whiten, generate_synthetic_data


def run_experiment_vs_d(
    d_values,
    n_fixed=5000,
    seeds=(0, 1, 2),
    n_iter_stochastic=1000,
    batch_size=64,
    lr_sgd=0.01,
    lr_adam=0.0005,
):
    rows = []

    for d_cur in d_values:
        source_types = ["laplace"] * d_cur
        for sd in seeds:
            print(f"[vs_D] D={d_cur}, N={n_fixed}, seed={sd}", flush=True)

            x_cur, _, a_cur = generate_synthetic_data(
                D=d_cur,
                N=n_fixed,
                source_types=source_types,
                seed=sd,
            )
            xw_cur, _, w_white_cur = center_whiten(x_cur)

            # FastICA baseline
            t0 = time.perf_counter()
            v_fast_cur, _, _, _, _ = fastica_unmixing_matrix(
                x_cur,
                whitening_fn=center_whiten,
                n_components=d_cur,
                max_iter=1000,
                tol=1e-6,
                seed=sd,
            )
            t_fast = time.perf_counter() - t0
            amari_fast = amari_index(v_fast_cur @ a_cur)
            rows.append(
                {
                    "experiment": "vs_D",
                    "D": d_cur,
                    "N": n_fixed,
                    "seed": sd,
                    "algo": "FastICA",
                    "amari": amari_fast,
                    "time_s": t_fast,
                    "n_iter": 1000,
                    "batch_size": np.nan,
                    "lr": np.nan,
                }
            )

            # SGD-ICA
            t0 = time.perf_counter()
            w_sgd_cur, _ = sgd_ica(
                xw_cur,
                n_iter=n_iter_stochastic,
                lr=lr_sgd,
                batch_size=batch_size,
                seed=sd,
            )
            t_sgd = time.perf_counter() - t0
            amari_sgd = amari_index((w_sgd_cur @ w_white_cur) @ a_cur)
            rows.append(
                {
                    "experiment": "vs_D",
                    "D": d_cur,
                    "N": n_fixed,
                    "seed": sd,
                    "algo": "SGD-ICA",
                    "amari": amari_sgd,
                    "time_s": t_sgd,
                    "n_iter": n_iter_stochastic,
                    "batch_size": batch_size,
                    "lr": lr_sgd,
                }
            )

            # Adam-ICA
            t0 = time.perf_counter()
            w_adam_cur, _ = adam_ica(
                xw_cur,
                n_iter=n_iter_stochastic,
                lr=lr_adam,
                batch_size=batch_size,
                seed=sd,
            )
            t_adam = time.perf_counter() - t0
            amari_adam = amari_index((w_adam_cur @ w_white_cur) @ a_cur)
            rows.append(
                {
                    "experiment": "vs_D",
                    "D": d_cur,
                    "N": n_fixed,
                    "seed": sd,
                    "algo": "Adam-ICA",
                    "amari": amari_adam,
                    "time_s": t_adam,
                    "n_iter": n_iter_stochastic,
                    "batch_size": batch_size,
                    "lr": lr_adam,
                }
            )

    return pd.DataFrame(rows)


def run_experiment_vs_n(
    n_values,
    d_fixed=20,
    seeds=(0, 1, 2),
    epochs_stochastic=5,
    batch_size=64,
    lr_sgd=0.01,
    lr_adam=0.001,
):
    rows = []
    source_types = ["laplace"] * d_fixed

    for n_cur in n_values:
        n_iter_cur = math.ceil(epochs_stochastic * n_cur / batch_size)
        for sd in seeds:
            print(f"[vs_N] D={d_fixed}, N={n_cur}, seed={sd}, n_iter={n_iter_cur}", flush=True)

            x_cur, _, a_cur = generate_synthetic_data(
                D=d_fixed,
                N=n_cur,
                source_types=source_types,
                seed=sd,
            )
            xw_cur, _, w_white_cur = center_whiten(x_cur)

            # FastICA baseline
            t0 = time.perf_counter()
            v_fast_cur, _, _, _, _ = fastica_unmixing_matrix(
                x_cur,
                whitening_fn=center_whiten,
                n_components=d_fixed,
                max_iter=1000,
                tol=1e-6,
                seed=sd,
            )
            t_fast = time.perf_counter() - t0
            amari_fast = amari_index(v_fast_cur @ a_cur)
            rows.append(
                {
                    "experiment": "vs_N",
                    "D": d_fixed,
                    "N": n_cur,
                    "seed": sd,
                    "algo": "FastICA",
                    "amari": amari_fast,
                    "time_s": t_fast,
                    "n_iter": 1000,
                    "batch_size": np.nan,
                    "lr": np.nan,
                }
            )

            # SGD-ICA
            t0 = time.perf_counter()
            w_sgd_cur, _ = sgd_ica(
                xw_cur,
                n_iter=n_iter_cur,
                lr=lr_sgd,
                batch_size=batch_size,
                seed=sd,
            )
            t_sgd = time.perf_counter() - t0
            amari_sgd = amari_index((w_sgd_cur @ w_white_cur) @ a_cur)
            rows.append(
                {
                    "experiment": "vs_N",
                    "D": d_fixed,
                    "N": n_cur,
                    "seed": sd,
                    "algo": "SGD-ICA",
                    "amari": amari_sgd,
                    "time_s": t_sgd,
                    "n_iter": n_iter_cur,
                    "batch_size": batch_size,
                    "lr": lr_sgd,
                }
            )

            # Adam-ICA
            t0 = time.perf_counter()
            w_adam_cur, _ = adam_ica(
                xw_cur,
                n_iter=n_iter_cur,
                lr=lr_adam,
                batch_size=batch_size,
                seed=sd,
            )
            t_adam = time.perf_counter() - t0
            amari_adam = amari_index((w_adam_cur @ w_white_cur) @ a_cur)
            rows.append(
                {
                    "experiment": "vs_N",
                    "D": d_fixed,
                    "N": n_cur,
                    "seed": sd,
                    "algo": "Adam-ICA",
                    "amari": amari_adam,
                    "time_s": t_adam,
                    "n_iter": n_iter_cur,
                    "batch_size": batch_size,
                    "lr": lr_adam,
                }
            )

    return pd.DataFrame(rows)


def aggregate_for_plot(df, x_col):
    return (
        df.groupby([x_col, "algo"]).agg(amari_mean=("amari", "mean"), amari_std=("amari", "std"), time_mean=("time_s", "mean"), time_std=("time_s", "std")).reset_index()
    )


def print_summary(df, x_col, title):
    print(f"\n=== {title} ===")
    x_values = sorted(df[x_col].unique())
    algos = ["FastICA", "SGD-ICA", "Adam-ICA"]

    for x in x_values:
        print(f"\n{x_col} = {x}")
        for a in algos:
            sub = df[(df[x_col] == x) & (df["algo"] == a)]
            am = sub["amari"].mean()
            ams = sub["amari"].std()
            tm = sub["time_s"].mean()
            tms = sub["time_s"].std()
            print(f"  {a:8s} | Amari={am:.4f} +/- {ams:.4f} | Time={tm:.3f}s +/- {tms:.3f}s")


def plot_pair(summary_df, x_col, x_label, title_quality, title_time, out_amari, out_time):
    algos = ["FastICA", "SGD-ICA", "Adam-ICA"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for a in algos:
        sub = summary_df[summary_df["algo"] == a].sort_values(x_col)
        ax.errorbar(
            sub[x_col],
            sub["amari_mean"],
            yerr=sub["amari_std"],
            marker="o",
            capsize=3,
            label=a,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Indice d'Amari (plus bas = mieux)")
    ax.set_title(title_quality)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_amari, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for a in algos:
        sub = summary_df[summary_df["algo"] == a].sort_values(x_col)
        ax.errorbar(
            sub[x_col],
            sub["time_mean"],
            yerr=sub["time_std"],
            marker="o",
            capsize=3,
            label=a,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Temps (s)")
    ax.set_title(title_time)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_time, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_all(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_d = run_experiment_vs_d(
        d_values=args.d_values,
        n_fixed=args.n_fixed,
        seeds=args.seeds,
        n_iter_stochastic=args.n_iter_stochastic,
        batch_size=args.batch_size,
        lr_sgd=args.lr_sgd_d,
        lr_adam=args.lr_adam_d,
    )

    df_n = run_experiment_vs_n(
        n_values=args.n_values,
        d_fixed=args.d_fixed,
        seeds=args.seeds,
        epochs_stochastic=args.epochs_stochastic,
        batch_size=args.batch_size,
        lr_sgd=args.lr_sgd_n,
        lr_adam=args.lr_adam_n,
    )

    df_all = pd.concat([df_d, df_n], ignore_index=True)

    df_d.to_csv(output_dir / "exp1_vs_d_results.csv", index=False)
    df_n.to_csv(output_dir / "exp1_vs_n_results.csv", index=False)
    df_all.to_csv(output_dir / "scalability_results.csv", index=False)

    summary_d = aggregate_for_plot(df_d, "D")
    summary_n = aggregate_for_plot(df_n, "N")

    print_summary(df_d, "D", "Moyennes par dimension")
    print_summary(df_n, "N", f"Scalabilite en N (D={args.d_fixed}, epochs={args.epochs_stochastic})")

    plot_pair(
        summary_df=summary_d,
        x_col="D",
        x_label="Dimension D",
        title_quality="Qualite de separation vs dimension",
        title_time="Cout de calcul vs dimension",
        out_amari=output_dir / "amari_vs_d.pdf",
        out_time=output_dir / "time_vs_d.pdf",
    )

    plot_pair(
        summary_df=summary_n,
        x_col="N",
        x_label="Nombre d echantillons N",
        title_quality="Qualite de separation vs N",
        title_time="Cout de calcul vs N",
        out_amari=output_dir / "amari_vs_n.pdf",
        out_time=output_dir / "time_vs_n.pdf",
    )

    print(f"\nSaved CSV and figures to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1 scalability analysis")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")

    parser.add_argument("--d_values", type=int, nargs="+", default=[3, 5, 10, 20, 40], help="Dimensions for vs_D experiment")
    parser.add_argument("--n_fixed", type=int, default=5000, help="Fixed N for vs_D experiment")
    parser.add_argument("--n_iter_stochastic", type=int, default=1000, help="Iterations for stochastic methods in vs_D")

    parser.add_argument("--n_values", type=int, nargs="+", default=[1000, 2000, 5000, 10000], help="Sample counts for vs_N experiment")
    parser.add_argument("--d_fixed", type=int, default=20, help="Fixed D for vs_N experiment")
    parser.add_argument("--epochs_stochastic", type=int, default=5, help="Fixed epochs for stochastic methods in vs_N")

    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for stochastic methods")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")

    parser.add_argument("--lr_sgd_d", type=float, default=0.01, help="SGD learning rate in vs_D")
    parser.add_argument("--lr_adam_d", type=float, default=0.0005, help="Adam learning rate in vs_D")
    parser.add_argument("--lr_sgd_n", type=float, default=0.01, help="SGD learning rate in vs_N")
    parser.add_argument("--lr_adam_n", type=float, default=0.001, help="Adam learning rate in vs_N")

    run_all(parser.parse_args())
