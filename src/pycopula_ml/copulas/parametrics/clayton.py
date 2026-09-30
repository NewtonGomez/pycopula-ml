"""
Bivariate Clayton copula.

For the bivariate absolutely continuous model implemented here, the
dependence parameter satisfies ``theta > -1``. The value ``theta = 0`` is
handled by :class:`ParametricBivariateCopula` as independence.

The CDF is

    C(u, v) = max(u**(-theta) + v**(-theta) - 1, 0)**(-1/theta)

for ``theta != 0``.
"""

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import ParametricBivariateCopula


class ClaytonCopula(ParametricBivariateCopula):
    """
    Bivariate Clayton copula.

    Notes
    -----
    ``theta = -1`` corresponds to the lower Fréchet bound, which is singular
    and does not possess an ordinary bivariate density. It is therefore
    excluded from this continuous-density implementation.
    """

    independence_parameter: ClassVar[float] = 0.0

    def _validate_theta(self) -> None:
        if self.theta <= -1.0:
            raise ValueError(
                "theta must be greater than -1 for the continuous "
                "bivariate Clayton copula."
            )

    def _cdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            base = (
                np.power(u, -self.theta)
                + np.power(v, -self.theta)
                - 1.0
            )

            return np.where(
                base > 0.0,
                np.power(base, -1.0 / self.theta),
                0.0,
            )

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
            log_u = np.log(u)
            log_v = np.log(v)

            base = (
                np.exp(-theta * log_u)
                + np.exp(-theta * log_v)
                - 1.0
            )

            result = np.full(
                base.shape,
                -np.inf,
                dtype=np.float64,
            )

            valid = base > 0.0

            result[valid] = (
                np.log1p(theta)
                - (theta + 1.0)
                * (log_u[valid] + log_v[valid])
                + (-2.0 - 1.0 / theta)
                * np.log(base[valid])
            )

        return result
