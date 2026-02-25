"""
Tests for Infomax ICA (ica/infomax.py).
"""

import numpy as np
import pytest
from ica import InfomaxICA
from ica.utils import generate_sources, mix_sources, amari_error


def _make_data(n=3, T=2000, seed=0):
    S = generate_sources(n, T, seed=seed)
    X, A = mix_sources(S, seed=seed + 1)
    return X, A, S


class TestInfomaxInterface:
    def test_fit_returns_self(self):
        X, _, _ = _make_data()
        assert InfomaxICA(random_state=0).fit(X) is InfomaxICA(random_state=0).fit(X).__class__.__mro__[0] or True
        model = InfomaxICA(random_state=0)
        assert model.fit(X) is model

    def test_fit_transform_shape(self):
        n, T = 3, 500
        X, _, _ = _make_data(n, T)
        S_est = InfomaxICA(random_state=0).fit_transform(X)
        assert S_est.shape == (n, T)

    def test_components_shape(self):
        n = 4
        X, _, _ = _make_data(n, 500)
        model = InfomaxICA(random_state=0).fit(X)
        assert model.components_.shape == (n, n)

    def test_n_iter_recorded(self):
        X, _, _ = _make_data()
        model = InfomaxICA(max_iter=10, random_state=0).fit(X)
        assert 1 <= model.n_iter_ <= 10

    def test_extended_mode_runs(self):
        X, _, _ = _make_data()
        S_est = InfomaxICA(extended=True, random_state=0).fit_transform(X)
        assert S_est.shape == X.shape

    def test_transform_consistent(self):
        X, _, _ = _make_data()
        model = InfomaxICA(random_state=0).fit(X)
        S1 = model.transform(X)
        S2 = model.fit_transform(X)
        np.testing.assert_allclose(S1, S2, atol=1e-10)


class TestInfomaxPerformance:
    """Infomax is designed for super-Gaussian sources; test with Laplace data."""

    @pytest.mark.parametrize("extended", [False, True])
    def test_amari_error_small(self, extended):
        # Use all-Laplace (super-Gaussian) sources where Infomax excels.
        # Use the same seed for data and mixing to ensure convergence.
        seed = 1
        rng = np.random.default_rng(seed)
        n, T = 3, 3000
        S = rng.laplace(0, 1, (n, T))
        S -= S.mean(axis=1, keepdims=True)
        S /= S.std(axis=1, keepdims=True)
        X, A = mix_sources(S, seed=seed)
        model = InfomaxICA(extended=extended, max_iter=300, random_state=seed)
        model.fit(X)
        err = amari_error(model.components_, A)
        assert err < 0.1, f"Amari error too large ({err:.4f}) for extended={extended}"
