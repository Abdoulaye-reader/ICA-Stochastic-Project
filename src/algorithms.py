import numpy as np
from sklearn.decomposition import FastICA


def amari_index(C):
	# Lower is better; 0 
	D = C.shape[0]
	absC = np.abs(C)

	row_max = absC.max(axis=1, keepdims=True)
	col_max = absC.max(axis=0, keepdims=True)

	row_term = (absC / (row_max + 1e-12)).sum(axis=1) - 1.0
	col_term = (absC / (col_max + 1e-12)).sum(axis=0) - 1.0

	return (row_term.sum() + col_term.sum()) / (2.0 * D * (D - 1))


def fastica_components(Xw, n_components=None, max_iter=1000, tol=1e-6, seed=42):
	if n_components is None:
		n_components = Xw.shape[0]

	fastica = FastICA(
		n_components=n_components,
		algorithm="parallel",
		fun="logcosh",
		whiten=False,
		max_iter=max_iter,
		tol=tol,
		random_state=seed,
	)
	fastica.fit(Xw.T)
	return fastica.components_


def fastica_unmixing_matrix(X, whitening_fn, n_components=None, max_iter=500, tol=1e-6, seed=42):
	# whitening_fn must return (Xw, mu, W_white) with Xw = W_white @ (X - mu).
	Xw, mu, W_white = whitening_fn(X)
	B = fastica_components(
		Xw,
		n_components=n_components,
		max_iter=max_iter,
		tol=tol,
		seed=seed,
	)
	# Map from whitened space back to original observation space.
	V_hat = B @ W_white
	return V_hat, B, Xw, mu, W_white
