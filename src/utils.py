import numpy as np


def center_whiten(X, eps=1e-8):
	"""Center and whiten data using PCA.
	
	Args:
		X: Input data (D x N)
		eps: Small constant for numerical stability
	
	Returns:
		Xw: Whitened data (D x N)
		mu: Mean vector (D x 1)
		W_white: Whitening matrix (D x D)
	"""
	mu = X.mean(axis=1, keepdims=True)
	Xc = X - mu

	cov = (Xc @ Xc.T) / Xc.shape[1]
	eigvals, eigvecs = np.linalg.eigh(cov)

	idx = np.argsort(eigvals)[::-1]
	eigvals = eigvals[idx]
	eigvecs = eigvecs[:, idx]

	D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
	W_white = D_inv_sqrt @ eigvecs.T
	Xw = W_white @ Xc

	return Xw, mu, W_white


def generate_synthetic_data(D=3, N=5000, source_types=None, mixing_rank=None, seed=42):
	"""Generate synthetic ICA problem with specified source distributions.
	
	Args:
		D: Number of sources
		N: Number of samples
		source_types: List of source types ('laplace', 'uniform', 'student', 'exp')
				If None, uses default ['laplace', 'uniform', 'student']
		mixing_rank: Rank of mixing matrix (default: D)
		seed: Random seed
	
	Returns:
		X: Mixed observations (D x N)
		S: True sources (D x N)
		A: Mixing matrix (D x D)
	"""
	if source_types is None:
		source_types = ['laplace', 'uniform', 'student']
	if mixing_rank is None:
		mixing_rank = D
	
	assert len(source_types) == D, "Number of source types must match D"
	
	rng = np.random.default_rng(seed)
	S = np.zeros((D, N))
	
	# Generate sources
	for i, stype in enumerate(source_types):
		if stype == 'laplace':
			S[i] = rng.laplace(loc=0, scale=1, size=N)
		elif stype == 'uniform':
			S[i] = rng.uniform(low=-1, high=1, size=N)
		elif stype == 'student':
			S[i] = rng.standard_t(df=3, size=N)
		elif stype == 'exp':
			S[i] = rng.exponential(scale=1, size=N) - 1
		else:
			raise ValueError(f"Unknown source type: {stype}")
	
	# Generate random mixing matrix
	A = rng.standard_normal((D, mixing_rank))
	if mixing_rank == D:
		A = rng.standard_normal((D, D))
	else:
		# For rank-deficient case, use full rank + projection
		U, _, Vt = np.linalg.svd(A, full_matrices=False)
		A = U @ np.diag(rng.uniform(0.5, 2, mixing_rank)) @ Vt
	
	X = A @ S
	
	return X, S, A


def compute_performance_metrics(V_est, A, S_true=None):
	"""Compute ICA performance metrics.
	
	Args:
		V_est: Estimated unmixing matrix (D x D)
		A: True mixing matrix (D x D)
		S_true: True sources, optional for additional metrics
	
	Returns:
		dict: Contains 'amari' and optionally 'mse'
	"""
	from src.algorithms import amari_index
	
	C = V_est @ A
	amari = amari_index(C)
	
	result = {'amari': amari}
	
	if S_true is not None:
		S_est = V_est @ (A @ S_true)
		# Normalize for fair comparison
		S_norm = S_true / (np.linalg.norm(S_true, axis=1, keepdims=True) + 1e-12)
		S_est_norm = S_est / (np.linalg.norm(S_est, axis=1, keepdims=True) + 1e-12)
		mse = np.mean((S_norm - S_est_norm) ** 2)
		result['source_mse'] = mse
	
	return result


def unmixing_from_whitened(W, W_white):
	"""Convert whitened-space matrix to original observation space.
	
	Args:
		W: Mixing matrix in whitened space (D x D)
		W_white: Whitening matrix (D x D)
	
	Returns:
		V: Unmixing matrix in original space (D x D)
	"""
	return W @ W_white
