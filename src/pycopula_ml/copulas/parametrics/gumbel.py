"""
Bivariate Gumbel-Hougaard copula.

The Gumbel-Hougaard copula is an Archimedean extreme-value copula with
``theta >= 1``. The value ``theta = 1`` represents independence and is
handled by the parametric base class.
"""

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import ParametricBivariateCopula


class GumbelCopula(ParametricBivariateCopula):
    """Bivariate Gumbel-Hougaard copula."""

    independence_parameter: ClassVar[float] = 1.0

    def _validate_theta(self) -> None:
        if self.theta < 1.0:
            raise ValueError(
                "theta must be greater than or equal to 1 for the "
                "Gumbel copula."
            )

    def _cdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        theta = self.theta

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            x = -np.log(u)
            y = -np.log(v)

            base = (
                np.power(x, theta)
                + np.power(y, theta)
            )

            radial_term = np.power(
                base,
                1.0 / theta,
            )

            return np.exp(-radial_term)

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

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            x = -np.log(u)
            y = -np.log(v)

            base = (
                np.power(x, theta)
                + np.power(y, theta)
            )

            radial_term = np.power(
                base,
                1.0 / theta,
            )

            return (
                -radial_term
                + np.log(radial_term + theta - 1.0)
                + (1.0 / theta - 2.0)
                * np.log(base)
                + (theta - 1.0)
                * (np.log(x) + np.log(y))
                - np.log(u)
                - np.log(v)
            )
