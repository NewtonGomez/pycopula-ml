"""Tests for scalar maximum-likelihood copula estimation."""

import numpy as np
import pytest

from pycopula_ml.estimation import fit_copula_mle


class QuadraticCopula:
    """Simple deterministic likelihood used to test the optimizer."""

    target = 2.5

    def __init__(self, *, theta: float) -> None:
        self.theta = theta

    def log_likelihood(self, u: np.ndarray, v: np.ndarray) -> float:
        del u, v
        return -(self.theta - self.target) ** 2


def valid_data():
    """Return valid pseudo-observations."""
    u = np.array([0.1, 0.3, 0.6, 0.9])
    v = np.array([0.2, 0.4, 0.7, 0.8])
    return u, v


def test_fit_recovers_known_maximum() -> None:
    """The optimizer should recover the maximum of a known likelihood."""
    u, v = valid_data()

    result = fit_copula_mle(
        QuadraticCopula,
        u,
        v,
        bounds=(0.1, 5.0),
    )

    assert result.success
    assert result.theta == pytest.approx(2.5, abs=1e-6)
    assert result.log_likelihood == pytest.approx(0.0, abs=1e-12)
    assert result.n_obs == 4


def test_fit_searches_multiple_intervals_and_selects_best() -> None:
    """The best likelihood should be selected across disjoint intervals."""
    u, v = valid_data()

    result = fit_copula_mle(
        QuadraticCopula,
        u,
        v,
        bounds=((-5.0, -0.1), (0.1, 5.0)),
    )

    assert result.theta == pytest.approx(2.5, abs=1e-6)
    assert result.interval == (0.1, 5.0)
    assert len(result.interval_results) == 2


@pytest.mark.parametrize(
    "u,v",
    [
        (np.array([0.0, 0.5]), np.array([0.2, 0.8])),
        (np.array([1.0, 0.5]), np.array([0.2, 0.8])),
        (np.array([-0.1, 0.5]), np.array([0.2, 0.8])),
        (np.array([0.1, 1.1]), np.array([0.2, 0.8])),
    ],
)
def test_fit_rejects_values_outside_open_unit_interval(u, v) -> None:
    """Pseudo-observations must be strictly inside the unit interval."""
    with pytest.raises(ValueError):
        fit_copula_mle(QuadraticCopula, u, v, bounds=(0.1, 5.0))


def test_fit_rejects_mismatched_shapes() -> None:
    """Paired samples must have the same shape."""
    u = np.array([0.1, 0.2, 0.3])
    v = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="same shape"):
        fit_copula_mle(QuadraticCopula, u, v, bounds=(0.1, 5.0))


def test_fit_rejects_reversed_bounds() -> None:
    """The lower bound must be smaller than the upper bound."""
    u, v = valid_data()

    with pytest.raises(ValueError, match="lower parameter bound"):
        fit_copula_mle(QuadraticCopula, u, v, bounds=(5.0, 0.1))
