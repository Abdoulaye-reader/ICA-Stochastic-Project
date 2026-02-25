"""
ICA Algorithm Implementations.

Contains implementations of:
- Infomax with batch gradient (Bell & Sejnowski, 1995)
- FastICA (Hyvärinen, 1999)
- SGD-ICA (Stochastic Gradient Descent variant)
- Adam-ICA (Adam optimizer variant)
- Natural Gradient SGD (Amari, 1998)
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict
from abc import ABC, abstractmethod


class ICAAlgorithm(ABC):
    """Base class for ICA algorithms."""
    
    def __init__(self, n_components: int, random_state: int = 42):
        """
        Parameters
        ----------
        n_components : int
            Number of independent components
        random_state : int
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.W_ = None
        self.loss_history_ = []
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'ICAAlgorithm':
        """Fit the ICA model to data X."""
        pass
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate independent sources.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Estimated sources (n_samples, n_components)
        """
        if self.W_ is None:
            raise ValueError("Model not fitted yet")
        return X @ self.W_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return estimated sources."""
        self.fit(X)
        return self.transform(X)


class InfomaxBatch(ICAAlgorithm):
    """
    Infomax ICA with batch gradient descent.
    
    Reference: Bell, A.J., & Sejnowski, T.J. (1995). An information-maximization 
    approach to blind separation. Neural Computation, 7(6), 1129-1159.
    """
    
    def __init__(
        self,
        n_components: int,
        fun: str = "logcosh",
        g_fun: Optional[Callable] = None,
        max_iter: int = 500,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of components
        fun : str
            Nonlinearity: 'logcosh', 'exp', 'cube'
        g_fun : callable, optional
            Custom score function g(y) = d/dy log p(y)
        max_iter : int
            Maximum number of iterations
        learning_rate : float
            Learning rate
        tol : float
            Convergence tolerance
        random_state : int
            Random seed
        """
        super().__init__(n_components, random_state)
        self.fun = fun
        self.g_fun = g_fun
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        
    def _get_nonlinearity(self, fun: str) -> Tuple[Callable, Callable]:
        """Get nonlinearity function and its derivative."""
        if fun == "logcosh":
            def g(y):
                return np.tanh(y)
            def g_prime(y):
                return 1 - np.tanh(y) ** 2
        elif fun == "exp":
            def g(y):
                return y * np.exp(-y ** 2 / 2)
            def g_prime(y):
                return (1 - y ** 2) * np.exp(-y ** 2 / 2)
        elif fun == "cube":
            def g(y):
                return y ** 3
            def g_prime(y):
                return 3 * y ** 2
        else:
            raise ValueError(f"Unknown function: {fun}")
        
        return g, g_prime
    
    def fit(self, X: np.ndarray) -> 'InfomaxBatch':
        """
        Fit Infomax ICA using batch gradient descent.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features)
            
        Returns
        -------
        self
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize unmixing matrix randomly
        self.W_ = np.random.randn(self.n_components, n_features)
        
        # QR decomposition for orthogonalization
        Q, R = np.linalg.qr(self.W_.T)
        self.W_ = (R @ Q.T).T
        
        if self.g_fun is None:
            g, _ = self._get_nonlinearity(self.fun)
        else:
            g = self.g_fun
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            # Forward pass
            Y = X @ self.W_.T  # (n_samples, n_components)
            
            # Compute loss (negative log-likelihood)
            det_W = np.linalg.det(self.W_)
            if det_W > 0:
                loss = -np.mean(np.log(np.abs(det_W))) - np.mean(np.log(1 + np.exp(-2*Y)))
            else:
                loss = 1e10
            
            self.loss_history_.append(loss)
            
            # Gradient computation
            # ∂L/∂W = -W^{-T} + 2*g(Y)^T*X / n_samples
            g_Y = g(Y)  # (n_samples, n_components)
            
            try:
                W_inv_T = np.linalg.inv(self.W_.T)
            except np.linalg.LinAlgError:
                break
            
            grad = -W_inv_T + 2 * (g_Y.T @ X) / n_samples
            
            # Update W
            self.W_ = self.W_ - self.learning_rate * grad
            
            # Check convergence
            if iteration > 0:
                delta = np.abs(loss - self.loss_history_[-2])
                if delta < self.tol:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return self


class FastICA(ICAAlgorithm):
    """
    FastICA algorithm implementation.
    
    Reference: Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for 
    independent component analysis. IEEE Transactions on Neural Networks, 10(3), 626-634.
    """
    
    def __init__(
        self,
        n_components: int,
        fun: str = "logcosh",
        max_iter: int = 500,
        tol: float = 1e-5,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of components
        fun : str
            Nonlinearity: 'logcosh', 'exp', 'cube'
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        random_state : int
            Random seed
        """
        super().__init__(n_components, random_state)
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        
    def _get_nonlinearity(self, fun: str) -> Tuple[Callable, Callable]:
        """Get nonlinearity and its derivative."""
        if fun == "logcosh":
            def g(y):
                return np.tanh(y)
            def g_prime(y):
                return 1 - np.tanh(y) ** 2
        elif fun == "exp":
            def g(y):
                return y * np.exp(-y ** 2 / 2)
            def g_prime(y):
                return (1 - y ** 2) * np.exp(-y ** 2 / 2)
        elif fun == "cube":
            def g(y):
                return y ** 3
            def g_prime(y):
                return 3 * y ** 2
        else:
            raise ValueError(f"Unknown function: {fun}")
        
        return g, g_prime
    
    def fit(self, X: np.ndarray) -> 'FastICA':
        """
        Fit FastICA using fixed-point iteration.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features)
            
        Returns
        -------
        self
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Center data
        X = X - np.mean(X, axis=0)
        
        # Whiten data
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        Q = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-8)) @ eigenvectors.T
        X = X @ Q.T
        
        # Initialize W randomly and orthonormalize
        self.W_ = np.random.randn(self.n_components, self.n_components)
        self.W_, _ = np.linalg.qr(self.W_)
        
        g, g_prime = self._get_nonlinearity(self.fun)
        
        # Fixed-point iteration
        for iteration in range(self.max_iter):
            # Update W using fixed-point iteration
            Y = X @ self.W_.T
            
            # w <- E[X * g(w^T X)] - E[g'(w^T X)] * w
            Eg = np.mean(g(Y)[:, np.newaxis] * X, axis=0)  # (n_features,)
            Egp = np.mean(g_prime(Y))
            
            W_new = Eg[:, np.newaxis] - Egp * self.W_.T
            
            # Orthogonalize using Gram-Schmidt
            self.W_, _ = np.linalg.qr(W_new.T)
            
            # Check convergence
            converged = np.all(np.abs(np.sum(self.W_ ** 2, axis=1) - 1) < self.tol)
            if converged and iteration > 0:
                print(f"FastICA converged at iteration {iteration}")
                break
        
        return self


class SGDICA(ICAAlgorithm):
    """
    SGD-ICA: Stochastic Gradient Descent variant of Infomax ICA.
    
    Uses mini-batch stochastic gradient descent for scalability.
    """
    
    def __init__(
        self,
        n_components: int,
        fun: str = "logcosh",
        max_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of components
        fun : str
            Nonlinearity function
        max_epochs : int
            Maximum number of epochs
        batch_size : int
            Batch size for SGD
        learning_rate : float
            Learning rate
        tol : float
            Convergence tolerance
        random_state : int
            Random seed
        """
        super().__init__(n_components, random_state)
        self.fun = fun
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tol = tol
        
    def _get_nonlinearity(self, fun: str) -> Callable:
        """Get nonlinearity function."""
        if fun == "logcosh":
            return lambda y: np.tanh(y)
        elif fun == "exp":
            return lambda y: y * np.exp(-y ** 2 / 2)
        elif fun == "cube":
            return lambda y: y ** 3
        else:
            raise ValueError(f"Unknown function: {fun}")
    
    def fit(self, X: np.ndarray) -> 'SGDICA':
        """
        Fit SGD-ICA.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features)
            
        Returns
        -------
        self
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Center and whiten
        X = X - np.mean(X, axis=0)
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        Q = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-8)) @ eigenvectors.T
        X = X @ Q.T
        
        # Initialize W
        self.W_ = np.random.randn(self.n_components, n_features)
        self.W_, _ = np.linalg.qr(self.W_.T)
        self.W_ = self.W_.T
        
        g = self._get_nonlinearity(self.fun)
        
        # SGD training loop
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_samples)
                X_batch = X[indices[batch_start:batch_end]]
                
                # Forward pass
                Y = X_batch @ self.W_.T
                
                # Gradient
                try:
                    W_inv_T = np.linalg.inv(self.W_.T)
                except np.linalg.LinAlgError:
                    continue
                
                g_Y = g(Y)
                grad = -W_inv_T + 2 * (g_Y.T @ X_batch) / X_batch.shape[0]
                
                # Update W
                self.W_ = self.W_ - self.learning_rate * grad
                
                epoch_loss += np.linalg.norm(grad)
            
            self.loss_history_.append(epoch_loss / (n_samples // self.batch_size))
        
        return self


class AdamICA(ICAAlgorithm):
    """
    Adam-ICA: Infomax ICA with Adam optimizer.
    
    Reference: Kingma, D.P., & Ba, J. (2015). Adam: A method for stochastic optimization.
    """
    
    def __init__(
        self,
        n_components: int,
        fun: str = "logcosh",
        max_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of components
        fun : str
            Nonlinearity function
        max_epochs : int
            Maximum epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate (default 0.001 for Adam)
        beta1 : float
            Exponential decay rate for 1st moment
        beta2 : float
            Exponential decay rate for 2nd moment
        epsilon : float
            Small constant for numerical stability
        random_state : int
            Random seed
        """
        super().__init__(n_components, random_state)
        self.fun = fun
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def _get_nonlinearity(self, fun: str) -> Callable:
        """Get nonlinearity function."""
        if fun == "logcosh":
            return lambda y: np.tanh(y)
        elif fun == "exp":
            return lambda y: y * np.exp(-y ** 2 / 2)
        elif fun == "cube":
            return lambda y: y ** 3
        else:
            raise ValueError(f"Unknown function: {fun}")
    
    def fit(self, X: np.ndarray) -> 'AdamICA':
        """
        Fit Adam-ICA.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features)
            
        Returns
        -------
        self
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Center and whiten
        X = X - np.mean(X, axis=0)
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        Q = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-8)) @ eigenvectors.T
        X = X @ Q.T
        
        # Initialize W
        self.W_ = np.random.randn(self.n_components, n_features)
        self.W_, _ = np.linalg.qr(self.W_.T)
        self.W_ = self.W_.T
        
        g = self._get_nonlinearity(self.fun)
        
        # Adam optimizer state
        m = np.zeros_like(self.W_)  # First moment
        v = np.zeros_like(self.W_)  # Second moment
        t = 0  # Time step
        
        # Training loop
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)
            
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_samples)
                X_batch = X[indices[batch_start:batch_end]]
                
                Y = X_batch @ self.W_.T
                
                try:
                    W_inv_T = np.linalg.inv(self.W_.T)
                except np.linalg.LinAlgError:
                    continue
                
                g_Y = g(Y)
                grad = -W_inv_T + 2 * (g_Y.T @ X_batch) / X_batch.shape[0]
                
                # Adam update
                t += 1
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
                
                m_hat = m / (1 - self.beta1 ** t)
                v_hat = v / (1 - self.beta2 ** t)
                
                self.W_ = self.W_ - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            self.loss_history_.append(np.linalg.norm(grad))
        
        return self


class NaturalGradientICA(ICAAlgorithm):
    """
    Natural Gradient SGD for ICA.
    
    Reference: Amari, S. (1998). Natural gradient works efficiently in learning.
    Neural Computation, 10(2), 251-276.
    """
    
    def __init__(
        self,
        n_components: int,
        fun: str = "logcosh",
        max_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of components
        fun : str
            Nonlinearity function
        max_epochs : int
            Maximum epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        random_state : int
            Random seed
        """
        super().__init__(n_components, random_state)
        self.fun = fun
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def _get_nonlinearity(self, fun: str) -> Callable:
        """Get nonlinearity function."""
        if fun == "logcosh":
            return lambda y: np.tanh(y)
        elif fun == "exp":
            return lambda y: y * np.exp(-y ** 2 / 2)
        elif fun == "cube":
            return lambda y: y ** 3
        else:
            raise ValueError(f"Unknown function: {fun}")
    
    def fit(self, X: np.ndarray) -> 'NaturalGradientICA':
        """
        Fit Natural Gradient ICA.
        
        Update rule: V <- V + η(I + g(y)y^T)V
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_features)
            
        Returns
        -------
        self
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Center and whiten
        X = X - np.mean(X, axis=0)
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        Q = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-8)) @ eigenvectors.T
        X = X @ Q.T
        
        # Initialize V (unmixing matrix)
        self.W_ = np.random.randn(self.n_components, n_features)
        self.W_, _ = np.linalg.qr(self.W_.T)
        self.W_ = self.W_.T
        
        g = self._get_nonlinearity(self.fun)
        
        # Training loop
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)
            
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_samples)
                X_batch = X[indices[batch_start:batch_end]]
                
                # Forward pass
                Y = X_batch @ self.W_.T  # (batch_size, n_components)
                
                g_Y = g(Y)  # (batch_size, n_components)
                
                # Natural gradient update: V <- V + η(I + g(y)y^T)V
                # For each sample in batch, compute average
                for i in range(X_batch.shape[0]):
                    y_i = Y[i:i+1].T  # (n_components, 1)
                    g_yi = g_Y[i:i+1].T  # (n_components, 1)
                    
                    # (I + g(y)y^T)
                    term = np.eye(self.n_components) + g_yi @ y_i.T
                    
                    self.W_ = self.W_ + self.learning_rate * (term @ self.W_)
            
            self.loss_history_.append(0)
        
        return self
