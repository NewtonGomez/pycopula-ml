"""Unit tests for copula-based mutual information."""

import numpy as np
import pytest

from pycopula_ml.dependence import gaussian_mutual_information, mutual_information


class IndependentCopula:
    def logpdf(self, u, v):
        return np.zeros_like(np.broadcast_arrays(u, v)[0])


class FGMCopula:
    """Valid Farlie-Gumbel-Morgenstern density for |theta| <= 1."""

    def __init__(self, theta):
        self.theta = theta

    def logpdf(self, u, v):
        return np.log1p(self.theta * (1 - 2 * u) * (1 - 2 * v))


@pytest.mark.parametrize("rho", [0.0, 0.3, -0.7])
def test_gaussian_closed_form(rho):
    assert gaussian_mutual_information(rho) == pytest.approx(
        -0.5 * np.log(1.0 - rho * rho)
    )


def test_gaussian_bad_parameter():
    with pytest.raises(ValueError):
        gaussian_mutual_information(1.0)


def test_independence_zero_mi():
    assert mutual_information(IndependentCopula()) == pytest.approx(0.0, abs=1e-12)


def test_quadrature_positive_dependence_and_convergence():
    copula = FGMCopula(0.7)
    mi32 = mutual_information(copula, order=32)
    mi64 = mutual_information(copula, order=64)
    assert mi64 > 0
    assert mi64 == pytest.approx(mi32, abs=1e-8)


def test_invalid_quadrature_order():
    with pytest.raises(ValueError):
        mutual_information(IndependentCopula(), order=4)
