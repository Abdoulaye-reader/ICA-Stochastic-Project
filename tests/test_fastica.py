"""
Tests for FastICA (ica/fastica.py).
"""

import numpy as np
import pytest
from ica import FastICA
from ica.utils import generate_sources, mix_sources, amari_error


def _make_data(n=3, T=2000, seed=0):
    S = generate_sources(n, T, seed=seed)
    X, A = mix_sources(S, seed=seed + 1)
    return X, A, S


class TestFastICAInterface:
    def test_fit_returns_self(self):
        X, _, _ = _make_data()
        model = FastICA(random_state=0)
        assert model.fit(X) is model

    def test_fit_transform_shape(self):
        n, T = 3, 1000
        X, _, _ = _make_data(n, T)
        S_est = FastICA(random_state=0).fit_transform(X)
        assert S_est.shape == (n, T)

    def test_components_shape(self):
        n = 4
        X, _, _ = _make_data(n, 1000)
        model = FastICA(random_state=0).fit(X)
        assert model.components_.shape == (n, n)

    def test_n_iter_deflation(self):
        X, _, _ = _make_data()
        model = FastICA(algorithm="deflation", random_state=0).fit(X)
        assert isinstance(model.n_iter_, list)
        assert len(model.n_iter_) == X.shape[0]

    def test_n_iter_symmetric(self):
        X, _, _ = _make_data()
        model = FastICA(algorithm="symmetric", random_state=0).fit(X)
        assert isinstance(model.n_iter_, int)

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError):
            FastICA(algorithm="bad").fit(np.eye(3))

    def test_invalid_g_raises(self):
        with pytest.raises(ValueError):
            FastICA(g="bad").fit(np.eye(3))

    def test_transform_consistent(self):
        X, _, _ = _make_data()
        model = FastICA(random_state=0).fit(X)
        S1 = model.transform(X)
        S2 = model.fit_transform(X)
        np.testing.assert_allclose(S1, S2, atol=1e-10)


class TestFastICAPerformance:
    """Check that FastICA achieves a reasonable separation quality."""

    @pytest.mark.parametrize("algorithm", ["deflation", "symmetric"])
    @pytest.mark.parametrize("g", ["logcosh", "exp", "cube"])
    def test_amari_error_small(self, algorithm, g):
        X, A, _ = _make_data(n=3, T=3000, seed=1)
        model = FastICA(algorithm=algorithm, g=g, random_state=1)
        model.fit(X)
        err = amari_error(model.components_, A)
        assert err < 0.1, f"Amari error too large ({err:.4f}) for {algorithm}/{g}"


class TestFastICANoWhitening:
    def test_no_whitening_runs(self):
        X, _, _ = _make_data()
        model = FastICA(whiten_data=False, random_state=0)
        S_est = model.fit_transform(X)
        assert S_est.shape == X.shape
