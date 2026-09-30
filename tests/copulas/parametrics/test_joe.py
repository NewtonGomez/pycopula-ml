"""Unit tests for the bivariate Joe copula."""

import numpy as np
import pytest

from pycopula_lm.copulas import JoeCopula


def test_joe_known_values():
    """Check CDF and PDF against closed-form values for theta = 2."""
    copula = JoeCopula(theta=2.0)

    expected_cdf = 1.0 - np.sqrt(7.0) / 4.0
    expected_pdf = 23.0 / (7.0 * np.sqrt(7.0))

    assert copula.cdf(0.5, 0.5) == pytest.approx(expected_cdf)
    assert copula.pdf(0.5, 0.5) == pytest.approx(expected_pdf)


def test_joe_cdf_margins():
    copula = JoeCopula(theta=2.5)
    u = np.array([0.1, 0.4, 0.8])
    v = np.array([0.2, 0.7, 0.9])

    np.testing.assert_allclose(copula.cdf(u, 1.0), u)
    np.testing.assert_allclose(copula.cdf(1.0, v), v)


@pytest.mark.parametrize("theta", [0.0, 0.5, 0.999999])
def test_joe_rejects_invalid_theta(theta):
    with pytest.raises(ValueError):
        JoeCopula(theta=theta)
