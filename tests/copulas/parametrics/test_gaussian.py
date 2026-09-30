"""Unit tests for the bivariate Gaussian copula."""

import numpy as np
import pytest

from pycopula_lm.copulas import GaussianCopula


def test_gaussian_known_values_at_center():
    """Use exact bivariate-normal identities at u = v = 0.5."""
    rho = 0.5
    copula = GaussianCopula(theta=rho)

    expected_cdf = 1.0 / 3.0
    expected_pdf = 2.0 / np.sqrt(3.0)

    assert copula.cdf(0.5, 0.5) == pytest.approx(
        expected_cdf,
        abs=1e-10,
    )
    assert copula.pdf(0.5, 0.5) == pytest.approx(
        expected_pdf,
        rel=1e-12,
    )


def test_gaussian_rho_alias():
    copula = GaussianCopula(theta=-0.35)

    assert copula.rho == pytest.approx(-0.35)


def test_gaussian_cdf_margins():
    """A copula must satisfy C(u, 1) = u and C(1, v) = v."""
    copula = GaussianCopula(theta=0.7)
    u = np.array([0.1, 0.4, 0.8])
    v = np.array([0.2, 0.7, 0.9])

    np.testing.assert_allclose(copula.cdf(u, 1.0), u)
    np.testing.assert_allclose(copula.cdf(1.0, v), v)
    np.testing.assert_allclose(
        copula.cdf(u, 0.0),
        np.zeros_like(u),
    )
    np.testing.assert_allclose(
        copula.cdf(0.0, v),
        np.zeros_like(v),
    )


@pytest.mark.parametrize("theta", [-1.0, 1.0, -1.1, 1.1])
def test_gaussian_rejects_invalid_theta(theta):
    with pytest.raises(ValueError):
        GaussianCopula(theta=theta)
