"""Unit tests for the bivariate Ali-Mikhail-Haq copula."""

import pytest

from pycopula_lm.copulas import AliMikhailHaqCopula


def test_amh_known_values():
    """Check CDF and PDF against closed-form values."""
    copula = AliMikhailHaqCopula(theta=0.5)

    assert copula.cdf(0.5, 0.5) == pytest.approx(2.0 / 7.0)
    assert copula.pdf(0.5, 0.5) == pytest.approx(352.0 / 343.0)


def test_amh_accepts_lower_boundary():
    copula = AliMikhailHaqCopula(theta=-1.0)

    assert copula.theta == pytest.approx(-1.0)


@pytest.mark.parametrize("theta", [-1.000001, 1.0, 1.1])
def test_amh_rejects_invalid_theta(theta):
    with pytest.raises(ValueError):
        AliMikhailHaqCopula(theta=theta)
