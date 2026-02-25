"""
FastICA — Fixed-point Independent Component Analysis.

Reference
---------
Hyvärinen A. & Oja E. (2000). Independent component analysis: algorithms and
applications. *Neural Networks*, 13(4–5), 411–430.
"""

from __future__ import annotations

import numpy as np

from .utils import whiten


# ---------------------------------------------------------------------------
# Non-linearity functions  g(u) = dG/du  and  g'(u) = d²G/du²
# ---------------------------------------------------------------------------

def _g_logcosh(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """g(u) = tanh(u),  g'(u) = 1 - tanh²(u)."""
    tanh_u = np.tanh(u)
    return tanh_u, 1.0 - tanh_u ** 2


def _g_exp(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """g(u) = u * exp(-u²/2),  g'(u) = (1 - u²) * exp(-u²/2)."""
    exp_u = np.exp(-0.5 * u ** 2)
    return u * exp_u, (1.0 - u ** 2) * exp_u


def _g_cube(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """g(u) = u³,  g'(u) = 3u²."""
    return u ** 3, 3.0 * u ** 2


_G_FUNCS = {
    "logcosh": _g_logcosh,
    "exp":     _g_exp,
    "cube":    _g_cube,
}


class FastICA:
    """FastICA — fixed-point ICA algorithm.

    Parameters
    ----------
    n_components : int, optional
        Number of independent components to extract.  Defaults to the
        number of observed channels.
    algorithm : {'deflation', 'symmetric'}
        Extraction strategy.
        - ``'deflation'``  extracts components one at a time (sequential).
        - ``'symmetric'``  updates all components simultaneously.
    g : {'logcosh', 'exp', 'cube'}
        Non-linearity used in the fixed-point iteration.
        - ``'logcosh'`` (default) is robust for most distributions.
        - ``'exp'`` is better for super-Gaussian sources.
        - ``'cube'`` is useful for sub-Gaussian sources.
    max_iter : int
        Maximum number of fixed-point iterations.
    tol : float
        Convergence tolerance on the change of the un-mixing matrix.
    whiten_data : bool
        Whether to whiten the data before running the algorithm.
    random_state : int or None
        Seed for random initialisation.

    Attributes
    ----------
    W_ : ndarray, shape (n_components, n_components)
        Estimated un-mixing matrix (in the whitened space).
    components_ : ndarray, shape (n_components, n_features)
        Un-mixing matrix projected back to the original (unwhitened) space.
    n_iter_ : int or list of int
        Number of iterations taken (per component in deflation mode).
    """

    def __init__(
        self,
        n_components: int | None = None,
        algorithm: str = "deflation",
        g: str = "logcosh",
        max_iter: int = 200,
        tol: float = 1e-4,
        whiten_data: bool = True,
        random_state: int | None = 0,
    ):
        if algorithm not in ("deflation", "symmetric"):
            raise ValueError(f"algorithm must be 'deflation' or 'symmetric', got '{algorithm}'")
        if g not in _G_FUNCS:
            raise ValueError(f"g must be one of {list(_G_FUNCS)}, got '{g}'")

        self.n_components = n_components
        self.algorithm = algorithm
        self.g = g
        self.max_iter = max_iter
        self.tol = tol
        self.whiten_data = whiten_data
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FastICA":
        """Fit the ICA model on *X*.

        Parameters
        ----------
        X : ndarray, shape (n_features, n_samples)
            Observation matrix (rows = channels, columns = samples).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n_features, n_samples = X.shape

        n_comp = self.n_components or n_features
        rng = np.random.default_rng(self.random_state)

        # --- Whitening ---
        if self.whiten_data:
            Xw, W_white, self._mean = whiten(X)
        else:
            Xw = X.copy()
            W_white = np.eye(n_features)
            self._mean = np.zeros(n_features)

        g_func = _G_FUNCS[self.g]

        if self.algorithm == "deflation":
            W, n_iter = self._deflation(Xw, n_comp, n_samples, g_func, rng)
        else:
            W, n_iter = self._symmetric(Xw, n_comp, n_samples, g_func, rng)

        self.W_ = W
        self.n_iter_ = n_iter
        # Project back to original space
        self.components_ = W @ W_white
        self._W_white = W_white
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned un-mixing to new data.

        Parameters
        ----------
        X : ndarray, shape (n_features, n_samples)

        Returns
        -------
        S : ndarray, shape (n_components, n_samples)
        """
        X = np.asarray(X, dtype=float)
        X_c = X - self._mean[:, None]
        return self.components_ @ X_c

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return estimated sources.

        Parameters
        ----------
        X : ndarray, shape (n_features, n_samples)

        Returns
        -------
        S : ndarray, shape (n_components, n_samples)
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deflation(self, Xw, n_comp, n_samples, g_func, rng):
        """Extract components one at a time (deflation / sequential)."""
        n = Xw.shape[0]
        W = np.zeros((n_comp, n))
        n_iters = []

        for p in range(n_comp):
            w = rng.standard_normal(n)
            w /= np.linalg.norm(w) + 1e-12

            for it in range(1, self.max_iter + 1):
                u = w @ Xw                    # shape (n_samples,)
                gval, gprime = g_func(u)
                w_new = (Xw @ gval) / n_samples - gprime.mean() * w

                # Gram-Schmidt deflation
                for j in range(p):
                    w_new -= (w_new @ W[j]) * W[j]

                norm = np.linalg.norm(w_new)
                w_new /= norm + 1e-12

                # Convergence check: |w_new · w| ≈ 1
                delta = abs(abs(w_new @ w) - 1.0)
                w = w_new
                if delta < self.tol:
                    break

            W[p] = w
            n_iters.append(it)

        return W, n_iters

    def _symmetric(self, Xw, n_comp, n_samples, g_func, rng):
        """Update all components simultaneously (symmetric decorrelation)."""
        n = Xw.shape[0]
        W = rng.standard_normal((n_comp, n))
        W = _sym_decorr(W)

        for it in range(1, self.max_iter + 1):
            U = W @ Xw                        # (n_comp, n_samples)
            gval, gprime = g_func(U)
            W_new = (gval @ Xw.T) / n_samples - np.diag(gprime.mean(axis=1)) @ W
            W_new = _sym_decorr(W_new)

            # Check convergence: max off-diagonal of W_new @ W.T should → 0
            lim = max(abs(abs(np.diag(W_new @ W.T)) - 1))
            W = W_new
            if lim < self.tol:
                break

        return W, it


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym_decorr(W: np.ndarray) -> np.ndarray:
    """Symmetric decorrelation: W ← (W W^T)^{-1/2} W."""
    S, V = np.linalg.eigh(W @ W.T)
    S = np.clip(S, 1e-12, None)
    return (V * (1.0 / np.sqrt(S))) @ V.T @ W
