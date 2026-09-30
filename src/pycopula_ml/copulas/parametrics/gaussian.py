"""
Bivariate Gaussian copula.

The dependence parameter ``theta`` is the linear correlation coefficient
``rho`` of the latent standard bivariate normal distribution and must satisfy

    -1 < theta < 1.

The value ``theta = 0`` represents independence and is handled by the
parametric base class.
"""

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import multivariate_normal

from .base import ParametricBivariateCopula


class GaussianCopula(ParametricBivariateCopula):
    """
    Bivariate Gaussian copula.

    Notes
    -----
    The parameter ``theta`` is interpreted as the Gaussian correlation
    coefficient ``rho``. The CDF is evaluated numerically through SciPy's
    bivariate normal distribution.
    """

    independence_parameter: ClassVar[float] = 0.0

    def _validate_theta(self) -> None:
        if not -1.0 < self.theta < 1.0:
            raise ValueError(
                "theta must lie strictly between -1 and 1 for the "
                "Gaussian copula."
            )

    @property
    def rho(self) -> float:
        """Return the Gaussian correlation parameter."""
        return self.theta

    def _cdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        rho = self.theta

        u_flat = u.ravel()
        v_flat = v.ravel()
        result = np.empty_like(u_flat)

        zero_mask = (u_flat <= 0.0) | (v_flat <= 0.0)
        result[zero_mask] = 0.0

        u_one_mask = (
            (u_flat >= 1.0)
            & ~zero_mask
        )
        result[u_one_mask] = v_flat[u_one_mask]

        v_one_mask = (
            (v_flat >= 1.0)
            & ~zero_mask
            & ~u_one_mask
        )
        result[v_one_mask] = u_flat[v_one_mask]

        interior = ~(
            zero_mask
            | u_one_mask
            | v_one_mask
        )

        if np.any(interior):
            points = np.column_stack(
                (
                    ndtri(u_flat[interior]),
                    ndtri(v_flat[interior]),
                )
            )

            covariance = np.array(
                [
                    [1.0, rho],
                    [rho, 1.0],
                ],
                dtype=np.float64,
            )

            result[interior] = multivariate_normal.cdf(
                points,
                mean=np.zeros(2, dtype=np.float64),
                cov=covariance,
            )

        return result.reshape(u.shape)

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
        rho = self.theta
        one_minus_rho_squared = 1.0 - rho**2

        z_u = ndtri(u)
        z_v = ndtri(v)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            quadratic_term = (
                2.0 * rho * z_u * z_v
                - rho**2 * (z_u**2 + z_v**2)
            ) / (
                2.0 * one_minus_rho_squared
            )

            return (
                -0.5 * np.log(one_minus_rho_squared)
                + quadratic_term
            )
