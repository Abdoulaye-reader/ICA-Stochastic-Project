import numpy as np
from sklearn.decomposition import FastICA


def amari_index(C):
	# Lower is better; 0 means perfect separation up to scale/permutation.
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
		fun="logcosh", #non linearité
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


def sgd_ica(Xw, n_iter=1000, lr=0.01, batch_size=64, seed=42, fun='logcosh'):
	"""SGD-ICA on Infomax with mini-batch gradient ascent.
	
	Args:
		Xw: Pre-whitened data (D x N)
		n_iter: Number of iterations
		lr: Learning rate
		batch_size: Batch size for SGD
		seed: Random seed
		fun: Non-linearity function ('logcosh' only)
	
	Returns:
		W: Mixing matrix in whitened space (D x D)
		history: Loss history across iterations
	"""
	D = Xw.shape[0]
	rng = np.random.default_rng(seed)
	N = Xw.shape[1]
	
	# Initialize V as an orthogonal matrix in whitened space.
	W = rng.standard_normal((D, D))
	W, _ = np.linalg.qr(W)
	
	# Score g(y)=d/dy log p(y).
	# For logcosh super-Gaussian prior, g(y) = -tanh(y).
	if fun == 'logcosh':
		score = lambda y: -np.tanh(y)
		logp = lambda y: -np.log(np.cosh(y))
	else:
		raise ValueError(f"Unknown fun: {fun}")
	
	history = []
	
	for it in range(n_iter):
		# Random batch
		idx = rng.choice(N, size=min(batch_size, N), replace=False)
		Xb = Xw[:, idx]
		
		# Forward: y = Vx
		Z = W @ Xb
		score_Z = score(Z)

		# Mini-batch approximation of course formula:
		# V <- V + lr * (V + E[g(Vx)x^T]).
		grad = W + (score_Z @ Xb.T) / Xb.shape[1]
		W = W + lr * grad

		# Practical note: keep V close to orthogonal after whitening.
		W, _ = np.linalg.qr(W)
		
		# Track Infomax proxy over full data.
		Z_full = W @ Xw
		sign, logabsdet = np.linalg.slogdet(W)
		ll = (logabsdet if sign != 0 else -np.inf) + np.mean(logp(Z_full))
		history.append(ll)
	
	return W, history


def adam_ica(Xw, n_iter=1000, lr=0.001, batch_size=64, seed=42, fun='logcosh', 
             beta1=0.9, beta2=0.999, eps=1e-8):
	"""Adam-ICA: Adam optimizer on the same Infomax gradient as SGD-ICA.
	
	Args:
		Xw: Pre-whitened data (D x N)
		n_iter: Number of iterations
		lr: Learning rate
		batch_size: Batch size for SGD
		seed: Random seed
		fun: Non-linearity function ('logcosh' only)
		beta1: Exponential decay rate for 1st moment
		beta2: Exponential decay rate for 2nd moment
		eps: Small constant for numerical stability
	
	Returns:
		W: Mixing matrix in whitened space (D x D)
		history: Loss history across iterations
	"""
	D = Xw.shape[0]
	rng = np.random.default_rng(seed)
	N = Xw.shape[1]
	
	# Initialize V as orthogonal in whitened space.
	W = rng.standard_normal((D, D))
	W, _ = np.linalg.qr(W)
	
	# Same score convention as sgd_ica.
	if fun == 'logcosh':
		score = lambda y: -np.tanh(y)
		logp = lambda y: -np.log(np.cosh(y))
	else:
		raise ValueError(f"Unknown fun: {fun}")
	
	# Adam state
	m = np.zeros((D, D))  # first moment
	v = np.zeros((D, D))  # second moment
	
	history = []
	
	for it in range(n_iter):
		# Random batch
		idx = rng.choice(N, size=min(batch_size, N), replace=False)
		Xb = Xw[:, idx]
		
		# Forward: y = Vx
		Z = W @ Xb
		score_Z = score(Z)

		# Same mini-batch gradient as SGD-ICA.
		grad = W + (score_Z @ Xb.T) / Xb.shape[1]
		
		# Adam update
		m = beta1 * m + (1 - beta1) * grad
		v = beta2 * v + (1 - beta2) * (grad ** 2)
		
		# Bias correction
		m_hat = m / (1 - beta1 ** (it + 1))
		v_hat = v / (1 - beta2 ** (it + 1))
		
		# Adam step in ascent mode.
		W = W + lr * m_hat / (np.sqrt(v_hat) + eps)

		# Keep V close to orthogonal after each update.
		W, _ = np.linalg.qr(W)
		
		# Track Infomax proxy over full data.
		Z_full = W @ Xw
		sign, logabsdet = np.linalg.slogdet(W)
		ll = (logabsdet if sign != 0 else -np.inf) + np.mean(logp(Z_full))
		history.append(ll)
	
	return W, history
