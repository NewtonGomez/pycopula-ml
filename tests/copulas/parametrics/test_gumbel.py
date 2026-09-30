"""Unit tests for the bivariate Gumbel-Hougaard copula."""

import numpy as np
import pytest

from pycopula_lm.copulas import GumbelCopula


def test_gumbel_known_values():
    """Check CDF and PDF at u = v = exp(-1), theta = 2."""
    copula = GumbelCopula(theta=2.0)
    u = np.exp(-1.0)

    expected_cdf = np.exp(-np.sqrt(2.0))
    expected_pdf = (
        np.exp(2.0 - np.sqrt(2.0))
        * (np.sqrt(2.0) + 1.0)
        / (2.0 ** 1.5)
    )

    assert copula.cdf(u, u) == pytest.approx(expected_cdf)
    assert copula.pdf(u, u) == pytest.approx(expected_pdf)


def test_gumbel_cdf_margins():
    copula = GumbelCopula(theta=2.5)
    u = np.array([0.1, 0.4, 0.8])
    v = np.array([0.2, 0.7, 0.9])

    np.testing.assert_allclose(copula.cdf(u, 1.0), u)
    np.testing.assert_allclose(copula.cdf(1.0, v), v)


@pytest.mark.parametrize("theta", [0.0, 0.5, 0.999999])
def test_gumbel_rejects_invalid_theta(theta):
    with pytest.raises(ValueError):
        GumbelCopula(theta=theta)
