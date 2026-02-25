"""
Tests for ica/utils.py.
"""

import numpy as np
import pytest
from ica.utils import (
    generate_sources,
    mix_sources,
    whiten,
    amari_error,
    signal_to_interference_ratio,
)


class TestGenerateSources:
    def test_shape(self):
        S = generate_sources(3, 500)
        assert S.shape == (3, 500)

    def test_zero_mean(self):
        S = generate_sources(4, 1000, seed=0)
        np.testing.assert_allclose(S.mean(axis=1), 0.0, atol=1e-10)

    def test_unit_variance(self):
        S = generate_sources(4, 1000, seed=0)
        np.testing.assert_allclose(S.std(axis=1), 1.0, atol=1e-10)

    def test_reproducibility(self):
        S1 = generate_sources(3, 200, seed=7)
        S2 = generate_sources(3, 200, seed=7)
        np.testing.assert_array_equal(S1, S2)

    def test_different_seeds(self):
        S1 = generate_sources(3, 200, seed=7)
        S2 = generate_sources(3, 200, seed=8)
        assert not np.allclose(S1, S2)


class TestMixSources:
    def test_shape(self):
        S = generate_sources(3, 300)
        X, A = mix_sources(S)
        assert X.shape == S.shape
        assert A.shape == (3, 3)

    def test_mixing_equation(self):
        S = generate_sources(3, 300, seed=0)
        X, A = mix_sources(S, seed=1)
        np.testing.assert_allclose(X, A @ S)

    def test_invertible_mixing(self):
        S = generate_sources(4, 300, seed=0)
        _, A = mix_sources(S, seed=1)
        # A must be invertible (rank == n)
        assert np.linalg.matrix_rank(A) == 4


class TestWhiten:
    def test_output_shape(self):
        X = np.random.default_rng(0).standard_normal((3, 500))
        Xw, W, mean = whiten(X)
        assert Xw.shape == X.shape
        assert W.shape == (3, 3)
        assert mean.shape == (3,)

    def test_whitened_covariance(self):
        X = np.random.default_rng(0).standard_normal((3, 5000))
        Xw, _, _ = whiten(X)
        cov = Xw @ Xw.T / Xw.shape[1]
        np.testing.assert_allclose(cov, np.eye(3), atol=0.05)

    def test_zero_mean_output(self):
        X = np.random.default_rng(0).standard_normal((3, 500)) + 5
        Xw, _, _ = whiten(X)
        np.testing.assert_allclose(Xw.mean(axis=1), 0.0, atol=1e-10)


class TestAmariError:
    def test_identity_gives_zero(self):
        n = 4
        A = np.eye(n)
        W = np.eye(n)
        err = amari_error(W, A)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_perfect_unmixing(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4))
        W = np.linalg.inv(A)
        err = amari_error(W, A)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_bad_unmixing_higher(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4))
        W = rng.standard_normal((4, 4))  # random unmixing
        W_perfect = np.linalg.inv(A)
        assert amari_error(W, A) > amari_error(W_perfect, A)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            A = rng.standard_normal((3, 3))
            W = rng.standard_normal((3, 3))
            assert amari_error(W, A) >= 0.0


class TestSIR:
    def test_perfect_separation(self):
        S = generate_sources(3, 1000, seed=0)
        # Perfect separation (up to scaling)
        sir = signal_to_interference_ratio(S * 2, S)
        assert np.all(sir > 30), f"SIR should be high, got {sir}"

    def test_returns_correct_shape(self):
        S = generate_sources(3, 500, seed=0)
        sir = signal_to_interference_ratio(S, S)
        assert sir.shape == (3,)
