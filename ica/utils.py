"""
Utility functions shared by all ICA algorithms.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def generate_sources(n_sources: int, n_samples: int, seed: int = 0) -> np.ndarray:
    """Generate *n_sources* independent non-Gaussian source signals.

    Returns a matrix of shape ``(n_sources, n_samples)`` whose rows are
    zero-mean, unit-variance independent sources drawn from various
    non-Gaussian distributions:

    * Sub-Gaussian  : uniform distribution
    * Super-Gaussian: Laplace distribution
    * Mixed         : sine wave (for even indices) and sawtooth (for odd)

    Parameters
    ----------
    n_sources : int
        Number of independent sources.
    n_samples : int
        Number of time samples per source.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray, shape (n_sources, n_samples)
        Source matrix with zero-mean, unit-variance rows.
    """
    rng = np.random.default_rng(seed)
    S = np.zeros((n_sources, n_samples))
    t = np.linspace(0, 2 * np.pi, n_samples)

    for i in range(n_sources):
        kind = i % 4
        if kind == 0:
            # sub-Gaussian: uniform
            s = rng.uniform(-1, 1, n_samples)
        elif kind == 1:
            # super-Gaussian: Laplace
            s = rng.laplace(0, 1, n_samples)
        elif kind == 2:
            # sinusoidal (periodic)
            freq = 1 + i // 4
            s = np.sin(freq * t)
        else:
            # sawtooth
            freq = 1 + i // 4
            s = 2 * ((freq * t / (2 * np.pi)) % 1) - 1

        # Standardise to zero mean, unit variance
        s -= s.mean()
        s /= s.std() + 1e-12
        S[i] = s

    return S


def mix_sources(S: np.ndarray, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Create a random full-rank mixing matrix and return mixed signals.

    Parameters
    ----------
    S : ndarray, shape (n_sources, n_samples)
        Source matrix.
    seed : int, optional
        Random seed for the mixing matrix.

    Returns
    -------
    X : ndarray, shape (n_sources, n_samples)
        Mixed observations (same shape as *S*).
    A : ndarray, shape (n_sources, n_sources)
        Mixing matrix used.
    """
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    A = rng.standard_normal((n, n))
    # Ensure A is invertible (condition number < 100)
    U, sv, Vt = np.linalg.svd(A)
    sv = np.clip(sv, 0.5, None)
    A = U @ np.diag(sv) @ Vt
    X = A @ S
    return X, A


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def whiten(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whiten (sphere) the data matrix *X*.

    Centres and then applies PCA whitening so that the output has identity
    covariance.

    Parameters
    ----------
    X : ndarray, shape (n_components, n_samples)
        Observation matrix (each row is one channel).

    Returns
    -------
    X_white : ndarray, shape (n_components, n_samples)
        Whitened observations.
    W_white : ndarray, shape (n_components, n_components)
        Whitening matrix such that ``X_white = W_white @ X_centred``.
    mean : ndarray, shape (n_components,)
        Column means subtracted during centering.
    """
    mean = X.mean(axis=1, keepdims=True)
    X_c = X - mean

    cov = X_c @ X_c.T / X_c.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Whitening matrix: D^{-1/2} * E^T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-12))
    W_white = D_inv_sqrt @ eigenvectors.T
    X_white = W_white @ X_c
    return X_white, W_white, mean.squeeze()


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def amari_error(W_est: np.ndarray, A_true: np.ndarray) -> float:
    """Compute the Amari performance index between estimated and true matrices.

    The Amari error is a permutation- and scaling-invariant measure of how
    well ``W_est`` (un-mixing matrix) recovers the original sources from
    ``A_true`` (mixing matrix).  A value of 0 indicates perfect separation.

    Parameters
    ----------
    W_est : ndarray, shape (n, n)
        Estimated unmixing matrix.
    A_true : ndarray, shape (n, n)
        True mixing matrix.

    Returns
    -------
    error : float
        Amari performance index in [0, 1).
    """
    P = W_est @ A_true
    n = P.shape[0]

    def _row_col_error(M):
        row_max = np.abs(M).max(axis=1, keepdims=True)
        col_max = np.abs(M).max(axis=0, keepdims=True)
        row_term = (np.abs(M) / (row_max + 1e-12)).sum(axis=1) - 1
        col_term = (np.abs(M) / (col_max + 1e-12)).sum(axis=0) - 1
        return row_term.sum() + col_term.sum()

    return _row_col_error(P) / (2 * n * (n - 1))


def signal_to_interference_ratio(S_est: np.ndarray, S_true: np.ndarray) -> np.ndarray:
    """Compute per-source Signal-to-Interference Ratio (SIR) in dB.

    Matches estimated sources to true sources by maximum absolute
    correlation (handles permutation ambiguity).

    Parameters
    ----------
    S_est : ndarray, shape (n, T)
        Estimated source matrix.
    S_true : ndarray, shape (n, T)
        True source matrix.

    Returns
    -------
    sir : ndarray, shape (n,)
        Per-source SIR values in dB.
    """
    n = S_true.shape[0]
    # Build absolute correlation matrix
    corr = np.abs(S_est @ S_true.T) / (
        np.linalg.norm(S_est, axis=1, keepdims=True) * np.linalg.norm(S_true, axis=1) + 1e-12
    )
    # Greedy matching (good enough for n <= 20)
    assigned = set()
    order = []
    for _ in range(n):
        best_i, best_j = -1, -1
        best_val = -1.0
        for i in range(n):
            if i in assigned:
                continue
            for j in range(n):
                if j in [o[1] for o in order]:
                    continue
                if corr[i, j] > best_val:
                    best_val = corr[i, j]
                    best_i, best_j = i, j
        assigned.add(best_i)
        order.append((best_i, best_j))

    sir = np.zeros(n)
    for est_idx, true_idx in order:
        s_true = S_true[true_idx]
        s_est = S_est[est_idx]
        # Scale s_est to match s_true
        scale = np.dot(s_est, s_true) / (np.dot(s_est, s_est) + 1e-12)
        s_signal = scale * s_est
        s_noise = s_true - s_signal
        power_signal = np.dot(s_signal, s_signal)
        power_noise = np.dot(s_noise, s_noise)
        sir[true_idx] = 10 * np.log10(power_signal / (power_noise + 1e-12))

    return sir
