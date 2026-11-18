"""Tests for pairwise matrix construction using mocked MLE results."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycopula_ml.dependence import build_weight_matrices, weight_matrix


class FGMCopula:
    def __init__(self, theta):
        self.theta = theta

    def logpdf(self, u, v):
        return np.log1p(self.theta * (1 - 2 * u) * (1 - 2 * v))


def test_weight_matrices_are_symmetric(monkeypatch):
    monkeypatch.setattr(
        weight_matrix,
        "fit_copula_mle",
        lambda *_args, **_kwargs: SimpleNamespace(
            theta=0.7, log_likelihood=5.0, success=True
        ),
    )
    u = np.random.default_rng(42).uniform(0.1, 0.9, size=(30, 4))
    result = build_weight_matrices(
        u,
        FGMCopula,
        bounds=(-1.0, 1.0),
        feature_names=("X1", "X2", "X3", "X4"),
    )
    for matrix in (result.theta, result.log_likelihood, result.mutual_information):
        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 0.0)
    np.testing.assert_allclose(result.weights, result.mutual_information)
    assert np.all(result.mutual_information[np.triu_indices(4, 1)] > 0)


def test_rejects_boundary_pseudoobservations():
    with pytest.raises(ValueError):
        build_weight_matrices(
            np.array([[0.0, 0.5], [0.5, 1.0]]), FGMCopula, bounds=(-1.0, 1.0)
        )
