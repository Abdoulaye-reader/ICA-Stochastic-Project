"""
Utility functions for ICA experiments.
Includes data generation, metrics, and helper functions.
"""

import numpy as np
from typing import Tuple, Dict


def generate_synthetic_data(
    n_samples: int = 1000,
    n_sources: int = 2,
    source_type: str = "laplace",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic ICA data with known sources and mixing matrix.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_sources : int
        Number of independent sources
    source_type : str
        Type of source distribution: 'laplace', 'uniform', 'exponential', 'subgaussian'
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    X : np.ndarray
        Observed mixed signals (n_samples, n_sources)
    S : np.ndarray
        True independent sources (n_samples, n_sources)
    W : np.ndarray
        True mixing matrix (n_sources, n_sources)
    """
    np.random.seed(seed)
    
    # Generate independent sources
    if source_type == "laplace":
        S = np.random.laplace(loc=0, scale=1, size=(n_samples, n_sources))
    elif source_type == "uniform":
        S = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(n_samples, n_sources))
    elif source_type == "exponential":
        S = np.random.exponential(scale=1, size=(n_samples, n_sources)) - 1
    elif source_type == "subgaussian":
        S = np.random.uniform(size=(n_samples, n_sources))
    else:
        raise ValueError(f"Unknown source type: {source_type}")
    
    # Generate random mixing matrix
    W = np.random.randn(n_sources, n_sources)
    
    # Ensure mixing matrix is well-conditioned
    U, _, Vt = np.linalg.svd(W)
    W = U @ Vt  # Use unitary matrix for better conditioning
    
    # Mix the sources
    X = S @ W.T
    
    return X, S, W


def amari_index(C: np.ndarray) -> float:
    """
    Compute Amari Index for measuring ICA separation quality.
    
    The Amari index measures how much the performance matrix C = V*W
    deviates from a perfect permutation-scaling structure (ideal 0, worse ~1).
    
    Parameters
    ----------
    C : np.ndarray
        Performance matrix C = V*W where V is estimated unmixing matrix
        and W is true mixing matrix (both n x n)
        
    Returns
    -------
    float
        Amari index value (0 = perfect separation, 1 = worst case)
    """
    D = C.shape[0]
    
    # Normalize rows
    row_sum = np.sum(np.abs(C), axis=1)
    C_row_norm = np.abs(C) / row_sum[:, np.newaxis]
    
    # Normalize columns
    col_sum = np.sum(np.abs(C), axis=0)
    C_col_norm = np.abs(C) / col_sum[np.newaxis, :]
    
    # Compute Amari index
    row_term = np.sum(np.sum(C_row_norm, axis=1) - 1)
    col_term = np.sum(np.sum(C_col_norm, axis=0) - 1)
    
    amari = (row_term + col_term) / (2 * D * (D - 1))
    
    return amari


def centering(X: np.ndarray) -> np.ndarray:
    """
    Center the data (zero mean).
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
        
    Returns
    -------
    np.ndarray
        Centered data
    """
    return X - np.mean(X, axis=0)


def whitening(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Whiten the data using ZCA whitening.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
        
    Returns
    -------
    Z : np.ndarray
        Whitened data
    Q : np.ndarray
        Whitening matrix
    """
    # Covariance matrix
    cov = np.cov(X.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Whitening matrix (ZCA)
    Q = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-8)) @ eigenvectors.T
    
    Z = X @ Q.T
    
    return Z, Q


def log_sum_exp(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Numerically stable log-sum-exp computation.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int, optional
        Axis along which to compute
        
    Returns
    -------
    np.ndarray
        log(sum(exp(x)))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
