"""
Bivariate Frank copula.

This module implements the bivariate Frank copula, an Archimedean copula
parameterized by a single dependence parameter ``theta``.

The Frank copula separates the dependence structure between two continuous
random variables from their marginal distributions. It is defined on the
unit square [0, 1]^2.

For ``theta > 0``, the Frank copula represents positive dependence. For
``theta < 0``, it represents negative dependence. As ``theta`` approaches
zero, the copula converges to the independence copula.

The module provides the :class:`FrankCopula` class, which implements:

- cumulative distribution function (CDF),
- probability density function (PDF),
- log-probability density function (log-PDF).

The log-PDF is implemented directly rather than as ``log(pdf())`` to improve
numerical stability during maximum likelihood estimation.

References
----------
Balakrishnan, N., & Lai, C.-D. (2009).
Continuous Bivariate Distributions, Second Edition.
Springer.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import BivariateCopula


@dataclass
class FrankCopula(BivariateCopula):
    """
    Bivariate Frank copula.

    Parameters
    ----------
    theta : float
        Dependence parameter of the Frank copula. The parameter must be
        finite and different from zero.

    Attributes
    ----------
    theta : float
        Dependence parameter of the copula.
    """

    theta: float

    def __post_init__(self) -> None:
        """
        Validate and normalize the Frank dependence parameter.

        Raises
        ------
        TypeError
            If ``theta`` cannot be converted to a floating-point value.
        ValueError
            If ``theta`` is not finite.
        ValueError
            If ``theta`` is equal to zero.

        Notes
        -----
        The direct Frank copula parameterization is undefined at
        ``theta = 0``. Independence is obtained as the limiting case when
        ``theta`` approaches zero.
        """
        try:
            self.theta = float(self.theta)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "theta must be a real numeric value."
            ) from exc

        if not np.isfinite(self.theta):
            raise ValueError("theta must be finite.")

        if self.theta == 0.0:
            raise ValueError(
                "theta must be different from zero. "
                "Independence is obtained as theta approaches zero."
            )

    def cdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the Frank copula cumulative distribution function.

        Parameters
        ----------
        u : array_like
            Values of the first uniform marginal variable. Values must lie
            in the interval [0, 1].
        v : array_like
            Values of the second uniform marginal variable. Values must lie
            in the interval [0, 1].

        Returns
        -------
        numpy.ndarray
            Values of the Frank copula CDF evaluated at ``(u, v)``.
        """
        u_array, v_array = self._prepare_inputs(u, v)

        exp_u_minus_one = np.expm1(-self.theta * u_array)
        exp_v_minus_one = np.expm1(-self.theta * v_array)
        exp_theta_minus_one = np.expm1(-self.theta)

        ratio = (
            exp_u_minus_one
            * exp_v_minus_one
            / exp_theta_minus_one
        )

        return -(1.0 / self.theta) * np.log1p(ratio)

    def pdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the Frank copula probability density function.

        Parameters
        ----------
        u : array_like
            Values of the first uniform marginal variable. Values must lie
            in the interval [0, 1].
        v : array_like
            Values of the second uniform marginal variable. Values must lie
            in the interval [0, 1].

        Returns
        -------
        numpy.ndarray
            Values of the Frank copula density evaluated at ``(u, v)``.
        """
        u_array, v_array = self._prepare_inputs(u, v)

        one_minus_exp_theta = -np.expm1(-self.theta)
        one_minus_exp_u = -np.expm1(-self.theta * u_array)
        one_minus_exp_v = -np.expm1(-self.theta * v_array)

        numerator = (
            self.theta
            * one_minus_exp_theta
            * np.exp(-self.theta * (u_array + v_array))
        )

        denominator_base = (
            one_minus_exp_theta
            - one_minus_exp_u * one_minus_exp_v
        )

        denominator = denominator_base**2

        return numerator / denominator

    def logpdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the logarithm of the Frank copula density.

        Parameters
        ----------
        u : array_like
            Values of the first uniform marginal variable. Values must lie
            in the interval [0, 1].
        v : array_like
            Values of the second uniform marginal variable. Values must lie
            in the interval [0, 1].

        Returns
        -------
        numpy.ndarray
            Natural logarithm of the Frank copula density evaluated at
            ``(u, v)``.
        """
        u_array, v_array = self._prepare_inputs(u, v)

        one_minus_exp_theta = -np.expm1(-self.theta)
        one_minus_exp_u = -np.expm1(-self.theta * u_array)
        one_minus_exp_v = -np.expm1(-self.theta * v_array)

        denominator_base = (
            one_minus_exp_theta
            - one_minus_exp_u * one_minus_exp_v
        )

        log_numerator = (
            np.log(self.theta * one_minus_exp_theta)
            - self.theta * (u_array + v_array)
        )

        log_denominator = (
            2.0 * np.log(np.abs(denominator_base))
        )

        return log_numerator - log_denominator
