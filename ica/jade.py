"""
JADE — Joint Approximate Diagonalization of Eigenmatrices.

Reference
---------
Cardoso J.-F. & Souloumiac A. (1993). Blind beamforming for non-Gaussian
signals. *IEE Proceedings F — Radar and Signal Processing*, 140(6), 362–370.

Cardoso J.-F. (1999). High-order contrasts for independent component
analysis. *Neural Computation*, 11(1), 157–192.
"""

from __future__ import annotations

import numpy as np

from .utils import whiten


class JADE:
    """JADE ICA algorithm based on cumulant-tensor joint diagonalisation.

    Parameters
    ----------
    n_components : int, optional
        Number of independent components to extract.
    max_iter : int
        Maximum number of Jacobi sweeps.
    tol : float
        Convergence tolerance on the off-diagonal energy of cumulant matrices.
    whiten_data : bool
        Whether to whiten the data before fitting.

    Attributes
    ----------
    W_ : ndarray, shape (n_components, n_components)
        Un-mixing matrix in whitened space.
    components_ : ndarray, shape (n_components, n_features)
        Un-mixing matrix projected back to original space.
    n_iter_ : int
        Number of Jacobi sweeps performed.
    """

    def __init__(
        self,
        n_components: int | None = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        whiten_data: bool = True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.whiten_data = whiten_data

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "JADE":
        """Fit JADE on *X*.

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

        if self.whiten_data:
            Xw, W_white, self._mean = whiten(X)
            Xw = Xw[:n_comp]
            W_white = W_white[:n_comp]
        else:
            Xw = X.copy()
            W_white = np.eye(n_features)
            self._mean = np.zeros(n_features)

        # Step 1: compute cumulant matrices
        CM = self._cumulant_matrices(Xw, n_comp)

        # Step 2: joint diagonalisation via Jacobi sweeps
        W, n_iter = self._joint_diag(CM, n_comp)

        self.W_ = W
        self.n_iter_ = n_iter
        self.components_ = W @ W_white
        self._W_white = W_white
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned un-mixing.

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
    # Private helpers
    # ------------------------------------------------------------------

    def _cumulant_matrices(self, Xw: np.ndarray, n: int) -> np.ndarray:
        """Build the set of n² fourth-order cumulant matrices.

        Returns an array of shape ``(n*n, n, n)``.
        """
        T = Xw.shape[1]
        # Estimate cumulant matrices via the formula:
        #   Q_{ij} = (1/T) Σ_t y_i(t) y_j(t) Y(t) Y(t)^T  − I·δ_{ij} − e_i e_j^T − e_j e_i^T
        # We use a vectorised implementation.

        # Scale: Xw already has identity covariance (whitened)
        # CM[p, :, :] = (1/T) Σ_t <w_p, y_t>² y_t y_t^T − R_p
        # where R_p is the correction for the Gaussian part.

        CM = np.zeros((n * n, n, n))
        for i in range(n):
            for j in range(n):
                s = Xw[i] * Xw[j]          # (T,)
                Q = (Xw * s[None, :]) @ Xw.T / T   # (n, n)
                # Subtract Gaussian corrections:
                # cum(y_i,y_j,y_k,y_l) = E[y_i y_j y_k y_l]
                #   - δ_{ij} δ_{kl} - δ_{ik} δ_{jl} - δ_{il} δ_{jk}
                # In matrix form: Q_{ij} -= δ_{ij} I + e_i e_j^T + e_j e_i^T
                if i == j:
                    Q -= np.eye(n)
                Q[i, j] -= 1.0   # - (e_i e_j^T) at (i,j)
                Q[j, i] -= 1.0   # - (e_j e_i^T) at (j,i)
                CM[i * n + j] = Q

        return CM

    def _joint_diag(self, CM: np.ndarray, n: int) -> tuple[np.ndarray, int]:
        """Joint diagonalisation of a set of symmetric matrices.

        Uses the Jacobi (cyclic-by-row) algorithm. Returns the orthogonal
        matrix *V* that simultaneously diagonalises all matrices and the
        number of sweeps performed.
        """
        V = np.eye(n)
        n_iter = 0

        for sweep in range(self.max_iter):
            n_iter = sweep + 1
            changed = False

            for p in range(n - 1):
                for q in range(p + 1, n):
                    # Build 2×2 cost tensors
                    g = np.array([
                        [CM[k, p, p] - CM[k, q, q], CM[k, p, q] + CM[k, q, p]]
                        for k in range(CM.shape[0])
                    ])  # (n², 2)

                    # Compute rotation angle via eigendecomposition of the
                    # 2×2 symmetric matrix G^T G
                    G = g.T @ g          # (2, 2)
                    ton, toff = G[0, 0] - G[1, 1], G[0, 1]
                    theta = 0.5 * np.arctan2(2.0 * toff, ton + np.sqrt(ton**2 + 4 * toff**2) + 1e-15)

                    if abs(theta) < self.tol:
                        continue

                    changed = True
                    c, s = np.cos(theta), np.sin(theta)

                    # Rotate all cumulant matrices
                    for k in range(CM.shape[0]):
                        Ckp = CM[k, :, p].copy()
                        Ckq = CM[k, :, q].copy()
                        CM[k, :, p] = c * Ckp + s * Ckq
                        CM[k, :, q] = -s * Ckp + c * Ckq
                        Cpk = CM[k, p, :].copy()
                        Cqk = CM[k, q, :].copy()
                        CM[k, p, :] = c * Cpk + s * Cqk
                        CM[k, q, :] = -s * Cpk + c * Cqk

                    # Accumulate rotation
                    Vp = V[:, p].copy()
                    Vq = V[:, q].copy()
                    V[:, p] = c * Vp + s * Vq
                    V[:, q] = -s * Vp + c * Vq

            if not changed:
                break

        return V.T, n_iter  # rows of V^T are the un-mixing directions
