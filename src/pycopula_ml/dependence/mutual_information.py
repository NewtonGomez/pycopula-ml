"""
Copula-based mutual information for continuous bivariate distributions.

For a fitted copula C with density c, I(X;Y) = int_0^1 int_0^1
c(u,v) log c(u,v) du dv. Values are in nats.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from pycopula_ml.copulas.parametrics.gaussian import GaussianCopula


def gaussian_mutual_information(rho: float) -> float:
    """
    Return exact MI (nats) for a bivariate Gaussian copula.

    Parameters
    ----------
    rho : float
        Copula correlation parameter, strictly between -1 and 1.
    """
    rho = float(rho)
    if not np.isfinite(rho) or not (-1.0 < rho < 1.0):
        raise ValueError("rho must be finite and strictly between -1 and 1.")
    return float(-0.5 * np.log1p(-rho * rho))


def mutual_information(copula: object, *, order: int = 64) -> float:
    """
    Compute MI from a fitted continuous bivariate copula, in nats.

    Uses the Gaussian analytic formula when possible; otherwise integrates
    c(u,v)*log(c(u,v)) over (0,1)^2 by Gauss-Legendre quadrature.

    Parameters
    ----------
    copula : BivariateCopula
        Fitted copula object exposing ``logpdf(u, v)``. For GaussianCopula,
        it must expose ``theta`` as its correlation parameter.
    order : int, default 64
        Quadrature nodes per dimension. Computational cost is order**2.
        Increase for strongly tail-dependent or near-boundary families.

    Returns
    -------
    float
        Mutual information in nats (natural logarithm).

    Raises
    ------
    ValueError
        If quadrature order is invalid or the integral is substantially
        negative (an indication of numerical inaccuracy).
    FloatingPointError
        If density evaluation is not numerically finite on quadrature nodes.
    """
    if isinstance(copula, GaussianCopula):
        return gaussian_mutual_information(copula.theta)

    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 8:
        raise ValueError("order must be an integer greater than or equal to 8.")

    # Map Legendre nodes and weights from [-1,1] to [0,1].
    points, weights = leggauss(order)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    u, v = np.meshgrid(points, points, indexing="ij")

    log_density = np.asarray(copula.logpdf(u, v), dtype=float)
    try:
        log_density = np.broadcast_to(log_density, u.shape)
    except ValueError as exc:
        raise ValueError(
            "copula.logpdf must return broadcast-compatible values."
        ) from exc

    if np.any(np.isnan(log_density)) or np.any(np.isposinf(log_density)):
        raise FloatingPointError("Non-finite copula log-density on quadrature grid.")

    # x*log(x) -> 0 when x -> 0, including log-density == -inf.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        density = np.exp(log_density)
        integrand = np.where(density == 0.0, 0.0, density * log_density)

    if not np.all(np.isfinite(integrand)):
        raise FloatingPointError("Numerical overflow in c(u,v)*log(c(u,v)).")

    value = float(np.einsum("i,ij,j->", weights, integrand, weights))
    if value < -1e-8:
        raise ValueError(
            "Numerical MI is negative; check the density or increase quadrature order."
        )
    return max(0.0, value)  # Remove only rounding-level negative noise.
