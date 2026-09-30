"""
Bivariate Joe copula.

The Joe copula is an Archimedean family with ``theta >= 1``. The value
``theta = 1`` represents independence and is handled by the parametric base
class.
"""

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import ParametricBivariateCopula


class JoeCopula(ParametricBivariateCopula):
    """Bivariate Joe copula."""

    independence_parameter: ClassVar[float] = 1.0

    def _validate_theta(self) -> None:
        if self.theta < 1.0:
            raise ValueError(
                "theta must be greater than or equal to 1 for the "
                "Joe copula."
            )

    def _cdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        theta = self.theta

        one_minus_u = 1.0 - u
        one_minus_v = 1.0 - v

        a = np.power(one_minus_u, theta)
        b = np.power(one_minus_v, theta)
        base = a + b - a * b

        return 1.0 - np.power(base, 1.0 / theta)

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
        theta = self.theta

        one_minus_u = 1.0 - u
        one_minus_v = 1.0 - v

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            a = np.power(one_minus_u, theta)
            b = np.power(one_minus_v, theta)
            base = a + b - a * b

            correction = (
                theta * base
                + (theta - 1.0)
                * (1.0 - a)
                * (1.0 - b)
            )

            return (
                (theta - 1.0)
                * (
                    np.log(one_minus_u)
                    + np.log(one_minus_v)
                )
                + (1.0 / theta - 2.0)
                * np.log(base)
                + np.log(correction)
            )
