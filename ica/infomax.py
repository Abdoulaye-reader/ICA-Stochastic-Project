"""
Infomax ICA — Bell & Sejnowski (1995) natural-gradient algorithm.

Reference
---------
Bell A. J. & Sejnowski T. J. (1995). An information-maximization approach to
blind separation and blind deconvolution. *Neural Computation*, 7(6),
1129–1159.

Extended Infomax (sub- and super-Gaussian sources) following:
Lee T.-W., Girolami M. & Sejnowski T. J. (1999). Independent component
analysis using an extended infomax algorithm for mixed subgaussian and
supergaussian sources. *Neural Computation*, 11(2), 417–441.
"""

from __future__ import annotations

import numpy as np

from .utils import whiten


class InfomaxICA:
    """Infomax ICA using the natural-gradient ascent rule.

    Supports the original (super-Gaussian) formulation and the extended
    Infomax that can also handle sub-Gaussian sources.

    Parameters
    ----------
    n_components : int, optional
        Number of independent components. Defaults to the number of channels.
    extended : bool
        If ``True``, use Extended Infomax which estimates the sign (kurtosis)
        of each source to choose between super- and sub-Gaussian non-linearity.
    learning_rate : float
        Initial learning rate η.
    max_iter : int
        Maximum number of epochs (passes over all samples).
    tol : float
        Convergence tolerance on the Frobenius norm of the weight change.
    batch_size : int
        Mini-batch size. Use the full dataset when set to ``-1``.
    anneal_step : float
        Factor by which the learning rate is multiplied after each epoch
        (annealing).  Set to 1.0 to disable annealing.
    whiten_data : bool
        Whether to whiten the data before fitting.
    random_state : int or None
        Seed for random initialisation.

    Attributes
    ----------
    W_ : ndarray, shape (n_components, n_components)
        Estimated un-mixing matrix (in whitened space).
    components_ : ndarray, shape (n_components, n_features)
        Un-mixing matrix projected back to original space.
    n_iter_ : int
        Number of epochs until convergence.
    """

    def __init__(
        self,
        n_components: int | None = None,
        extended: bool = False,
        learning_rate: float = 0.1,
        max_iter: int = 200,
        tol: float = 1e-4,
        batch_size: int = -1,
        anneal_step: float = 1.0,
        whiten_data: bool = True,
        random_state: int | None = 0,
    ):
        self.n_components = n_components
        self.extended = extended
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.anneal_step = anneal_step
        self.whiten_data = whiten_data
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "InfomaxICA":
        """Fit the Infomax ICA model.

        Parameters
        ----------
        X : ndarray, shape (n_features, n_samples)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n_features, n_samples = X.shape
        n_comp = self.n_components or n_features
        rng = np.random.default_rng(self.random_state)

        if self.whiten_data:
            Xw, W_white, self._mean = whiten(X)
        else:
            Xw = X.copy()
            W_white = np.eye(n_features)
            self._mean = np.zeros(n_features)

        # Initialise un-mixing matrix as identity
        W = np.eye(n_comp) + 0.1 * rng.standard_normal((n_comp, n_comp))

        bs = self.batch_size if self.batch_size > 0 else n_samples
        lr = self.learning_rate

        # Running estimate of kurtosis sign for Extended Infomax
        # (+1 = super-Gaussian, -1 = sub-Gaussian)
        kurt_running = np.zeros(n_comp)

        n_iter = 0
        for epoch in range(self.max_iter):
            # Shuffle columns for stochastic updates
            perm = rng.permutation(n_samples)
            Xw_shuf = Xw[:, perm]
            W_prev = W.copy()

            for start in range(0, n_samples, bs):
                Xb = Xw_shuf[:, start: start + bs]
                nb = Xb.shape[1]

                Y = W @ Xb  # (n_comp, nb)
                # Clip to prevent overflow in non-linearities
                Y = np.clip(Y, -30.0, 30.0)

                if self.extended:
                    # Update running kurtosis estimate
                    k4 = np.mean(Y ** 4, axis=1)
                    k2 = np.mean(Y ** 2, axis=1)
                    kurt_batch = k4 - 3.0 * k2 ** 2
                    kurt_running = 0.9 * kurt_running + 0.1 * kurt_batch
                    signs = np.where(kurt_running >= 0, 1.0, -1.0)
                    # super-Gaussian: 1-2σ(y),  sub-Gaussian: tanh(y)
                    nonlin = np.where(
                        signs[:, None] > 0,
                        1.0 - 2.0 * _sigmoid(Y),
                        np.tanh(Y),
                    )
                else:
                    # Super-Gaussian: g(y) = 1 - 2*sigmoid(y)
                    nonlin = 1.0 - 2.0 * _sigmoid(Y)

                # Natural-gradient update rule:
                # ΔW = lr * (I + nonlin * Y^T) * W
                dW = (np.eye(n_comp) + (nonlin @ Y.T) / nb) @ W
                W_candidate = W + lr * dW
                # Guard against numerical blow-up
                if np.isfinite(W_candidate).all():
                    W = W_candidate
                    # Prevent unbounded growth that would cause overflow
                    w_norm = np.linalg.norm(W, "fro")
                    if w_norm > 1e6:
                        W /= w_norm / np.sqrt(n_comp)

            lr *= self.anneal_step
            n_iter = epoch + 1

            # Convergence check
            delta = np.linalg.norm(W - W_prev, "fro") / (np.linalg.norm(W_prev, "fro") + 1e-12)
            if delta < self.tol:
                break

        self.W_ = W
        self.n_iter_ = n_iter
        self.components_ = W @ W_white
        self._W_white = W_white
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned un-mixing matrix to *X*.

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
        """Fit and return estimated sources."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _update_signs(self, Y: np.ndarray, signs: np.ndarray) -> np.ndarray:
        """Update kurtosis signs for Extended Infomax (legacy helper)."""
        kurt = np.mean(Y ** 4, axis=1) - 3.0 * np.mean(Y ** 2, axis=1) ** 2
        new_signs = np.where(kurt > 0, 1.0, -1.0)
        return np.where(signs * new_signs > 0, new_signs, signs)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
