"""
Basic experiment comparing different ICA algorithms.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/abdoulayediallo/ICA/ICA-Stochastic-Project')

from src.utils import generate_synthetic_data, amari_index, centering, whitening
from src.algorithms import InfomaxBatch, FastICA, SGDICA, AdamICA, NaturalGradientICA


def run_experiment():
    """Run basic ICA experiment comparing multiple algorithms."""
    
    # Configuration
    n_samples = 5000
    n_sources = 3
    n_runs = 5
    
    algorithms = {
        'Infomax Batch': InfomaxBatch(
            n_components=n_sources,
            max_iter=200,
            learning_rate=0.01
        ),
        'FastICA': FastICA(
            n_components=n_sources,
            max_iter=200
        ),
        'SGD-ICA': SGDICA(
            n_components=n_sources,
            max_epochs=50,
            batch_size=64,
            learning_rate=0.01
        ),
        'Adam-ICA': AdamICA(
            n_components=n_sources,
            max_epochs=50,
            batch_size=64,
            learning_rate=0.001
        ),
    }
    
    results = {name: [] for name in algorithms}
    
    print("Running ICA Experiments...")
    print("=" * 60)
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        
        # Generate synthetic data
        X, S_true, W_true = generate_synthetic_data(
            n_samples=n_samples,
            n_sources=n_sources,
            source_type='laplace',
            seed=42 + run
        )
        
        # Preprocess
        X = centering(X)
        X, Q = whitening(X)
        
        for algo_name, algo in algorithms.items():
            try:
                # Fit algorithm
                algo.fit(X)
                
                # Get estimated unmixing matrix
                V = algo.W_
                
                # Compute performance matrix
                # V should be inverse of W (up to permutation and scaling)
                C = V @ W_true
                
                # Compute Amari index
                amari = amari_index(C)
                results[algo_name].append(amari)
                
                print(f"  {algo_name}: Amari Index = {amari:.6f}")
                
            except Exception as e:
                print(f"  {algo_name}: ERROR - {str(e)}")
                results[algo_name].append(np.nan)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (Amari Index - lower is better)")
    print("=" * 60)
    
    for algo_name in algorithms:
        amari_values = np.array(results[algo_name])
        mean_amari = np.nanmean(amari_values)
        std_amari = np.nanstd(amari_values)
        print(f"{algo_name:20s}: {mean_amari:.6f} ± {std_amari:.6f}")


if __name__ == '__main__':
    run_experiment()
