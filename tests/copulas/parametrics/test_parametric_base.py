"""Shared tests for parametric bivariate copula behavior."""

import numpy as np
import pytest

from pycopula_lm.copulas import (
    AliMikhailHaqCopula,
    ClaytonCopula,
    GaussianCopula,
    GumbelCopula,
    JoeCopula,
)


@pytest.mark.parametrize(
    ("copula_class", "theta"),
    [
        (ClaytonCopula, 0.0),
        (GaussianCopula, 0.0),
        (JoeCopula, 1.0),
        (GumbelCopula, 1.0),
        (AliMikhailHaqCopula, 0.0),
    ],
)
def test_independence_case(copula_class, theta):
    """All families must reduce to the independence copula."""
    u = np.array([0.1, 0.4, 0.8])
    v = np.array([0.2, 0.7, 0.9])

    copula = copula_class(theta=theta)

    np.testing.assert_allclose(copula.cdf(u, v), u * v)
    np.testing.assert_allclose(copula.pdf(u, v), np.ones_like(u))
    np.testing.assert_allclose(copula.logpdf(u, v), np.zeros_like(u))
    assert copula.log_likelihood(u, v) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("copula_class", "theta"),
    [
        (ClaytonCopula, 2.0),
        (GaussianCopula, 0.6),
        (JoeCopula, 2.0),
        (GumbelCopula, 2.0),
        (AliMikhailHaqCopula, 0.5),
    ],
)
def test_pdf_and_logpdf_are_consistent(copula_class, theta):
    """The analytical log-density must agree with log(pdf)."""
    u = np.array([0.2, 0.4, 0.7])
    v = np.array([0.3, 0.6, 0.8])

    copula = copula_class(theta=theta)
    pdf = copula.pdf(u, v)
    logpdf = copula.logpdf(u, v)

    assert np.all(np.isfinite(pdf))
    assert np.all(pdf > 0.0)
    np.testing.assert_allclose(
        logpdf,
        np.log(pdf),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("copula_class", "theta"),
    [
        (ClaytonCopula, 2.0),
        (GaussianCopula, -0.4),
        (JoeCopula, 2.0),
        (GumbelCopula, 2.0),
        (AliMikhailHaqCopula, -0.5),
    ],
)
def test_copula_is_symmetric(copula_class, theta):
    """The implemented bivariate families are exchangeable."""
    u = np.array([0.2, 0.4, 0.7])
    v = np.array([0.3, 0.6, 0.8])

    copula = copula_class(theta=theta)

    np.testing.assert_allclose(
        copula.cdf(u, v),
        copula.cdf(v, u),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        copula.pdf(u, v),
        copula.pdf(v, u),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("copula_class", "theta"),
    [
        (ClaytonCopula, 2.0),
        (GaussianCopula, 0.6),
        (JoeCopula, 2.0),
        (GumbelCopula, 2.0),
        (AliMikhailHaqCopula, 0.5),
    ],
)
def test_log_likelihood_is_sum_of_logpdf(copula_class, theta):
    """Log-likelihood must equal the sample sum of log-density values."""
    u = np.array([0.2, 0.4, 0.7])
    v = np.array([0.3, 0.6, 0.8])

    copula = copula_class(theta=theta)

    expected = np.sum(copula.logpdf(u, v))

    assert copula.log_likelihood(u, v) == pytest.approx(
        float(expected)
    )
