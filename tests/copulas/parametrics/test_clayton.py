"""Unit tests for the bivariate Clayton copula."""

import numpy as np
import pytest

from pycopula_lm.copulas import ClaytonCopula


def test_clayton_known_values():
    """Check CDF and PDF against closed-form values."""
    copula = ClaytonCopula(theta=1.0)

    cdf = copula.cdf(0.5, 0.5)
    pdf = copula.pdf(0.5, 0.5)

    assert cdf == pytest.approx(1.0 / 3.0)
    assert pdf == pytest.approx(32.0 / 27.0)


def test_clayton_negative_dependence_can_have_zero_density():
    """For negative theta, density is zero outside its support."""
    copula = ClaytonCopula(theta=-0.5)

    assert copula.cdf(0.1, 0.1) == pytest.approx(0.0)
    assert copula.pdf(0.1, 0.1) == pytest.approx(0.0)
    assert np.isneginf(copula.logpdf(0.1, 0.1))


@pytest.mark.parametrize("theta", [-1.0, -1.1, -2.0])
def test_clayton_rejects_invalid_theta(theta):
    """The continuous Clayton implementation requires theta > -1."""
    with pytest.raises(ValueError):
        ClaytonCopula(theta=theta)


def test_clayton_accepts_theta_close_to_lower_boundary():
    copula = ClaytonCopula(theta=-0.999)

    assert copula.theta == pytest.approx(-0.999)
