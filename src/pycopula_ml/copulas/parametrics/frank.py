"""
Bivariate Frank copula.

The Frank copula is an Archimedean copula with dependence parameter
``theta``. Positive and negative values represent positive and negative
dependence, respectively. ``theta = 0`` is handled by the parametric base
class as the independence copula.
"""

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import ParametricBivariateCopula


class FrankCopula(ParametricBivariateCopula):
    """Bivariate Frank copula."""

    independence_parameter: ClassVar[float] = 0.0

    def _cdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        exp_u_minus_one = np.expm1(-self.theta * u)
        exp_v_minus_one = np.expm1(-self.theta * v)
        exp_theta_minus_one = np.expm1(-self.theta)

        ratio = (
            exp_u_minus_one
            * exp_v_minus_one
            / exp_theta_minus_one
        )

        return -(1.0 / self.theta) * np.log1p(ratio)

    def _pdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.exp(self._logpdf(u, v))

    def _logpdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        one_minus_exp_theta = -np.expm1(-self.theta)
        one_minus_exp_u = -np.expm1(-self.theta * u)
        one_minus_exp_v = -np.expm1(-self.theta * v)

        denominator_base = (
            one_minus_exp_theta
            - one_minus_exp_u * one_minus_exp_v
        )

        log_numerator = (
            np.log(self.theta * one_minus_exp_theta)
            - self.theta * (u + v)
        )

        log_denominator = 2.0 * np.log(
            np.abs(denominator_base)
        )

        return log_numerator - log_denominator
