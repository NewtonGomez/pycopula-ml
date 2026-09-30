"""
Bivariate Ali-Mikhail-Haq copula.

The Ali-Mikhail-Haq (AMH) copula has dependence parameter

    -1 <= theta < 1.

The value ``theta = 0`` represents independence and is handled by the
parametric base class.
"""

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import ParametricBivariateCopula


class AliMikhailHaqCopula(ParametricBivariateCopula):
    """Bivariate Ali-Mikhail-Haq copula."""

    independence_parameter: ClassVar[float] = 0.0

    def _validate_theta(self) -> None:
        if self.theta < -1.0 or self.theta >= 1.0:
            raise ValueError(
                "theta must satisfy -1 <= theta < 1 for the "
                "Ali-Mikhail-Haq copula."
            )

    def _cdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        denominator = (
            1.0
            - self.theta
            * (1.0 - u)
            * (1.0 - v)
        )

        return u * v / denominator

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

        denominator = (
            1.0
            - theta
            * (1.0 - u)
            * (1.0 - v)
        )

        numerator = (
            1.0
            + theta
            * (
                (1.0 + u) * (1.0 + v)
                - 3.0
            )
            + theta**2
            * (1.0 - u)
            * (1.0 - v)
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            return (
                np.log(numerator)
                - 3.0 * np.log(denominator)
            )
