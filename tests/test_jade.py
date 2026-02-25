"""
Tests for JADE ICA (ica/jade.py).
"""

import numpy as np
import pytest
from ica import JADE
from ica.utils import generate_sources, mix_sources, amari_error


def _make_data(n=3, T=2000, seed=0):
    S = generate_sources(n, T, seed=seed)
    X, A = mix_sources(S, seed=seed + 1)
    return X, A, S


class TestJADEInterface:
    def test_fit_returns_self(self):
        X, _, _ = _make_data()
        model = JADE()
        assert model.fit(X) is model

    def test_fit_transform_shape(self):
        n, T = 3, 500
        X, _, _ = _make_data(n, T)
        S_est = JADE().fit_transform(X)
        assert S_est.shape == (n, T)

    def test_components_shape(self):
        n = 4
        X, _, _ = _make_data(n, 500)
        model = JADE().fit(X)
        assert model.components_.shape == (n, n)

    def test_n_iter_recorded(self):
        X, _, _ = _make_data()
        model = JADE(max_iter=50).fit(X)
        assert 1 <= model.n_iter_ <= 50

    def test_transform_consistent(self):
        X, _, _ = _make_data()
        model = JADE().fit(X)
        S1 = model.transform(X)
        S2 = model.fit_transform(X)
        np.testing.assert_allclose(S1, S2, atol=1e-10)


class TestJADEPerformance:
    def test_amari_error_small(self):
        X, A, _ = _make_data(n=3, T=3000, seed=3)
        model = JADE()
        model.fit(X)
        err = amari_error(model.components_, A)
        assert err < 0.15, f"Amari error too large ({err:.4f})"
