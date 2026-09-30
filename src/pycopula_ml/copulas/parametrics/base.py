"""
Base classes for parametric bivariate copulas.

This module extends the generic bivariate copula interface with common
behavior for one-parameter parametric copula families.

The public ``cdf``, ``pdf``, and ``logpdf`` methods implement the common
independence case and delegate family-specific calculations to ``_cdf``,
``_pdf``, and ``_logpdf``.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import BivariateCopula


@dataclass
class ParametricBivariateCopula(BivariateCopula):
    """
    Base class for one-parameter continuous bivariate copulas.

    Parameters
    ----------
    theta : float
        Dependence parameter of the copula family.

    Notes
    -----
    Subclasses define their independence parameter through
    :attr:`independence_parameter`. At independence, the public methods use

    .. math::

        C(u, v) = uv,

        c(u, v) = 1,

        log c(u, v) = 0.

    This avoids evaluating family-specific formulas at parameter values where
    their direct expressions may be undefined even though the limiting copula
    is well defined.
    """

    theta: float

    independence_parameter: ClassVar[float] = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize the dependence parameter."""
        try:
            self.theta = float(self.theta)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "theta must be a real numeric value."
            ) from exc

        if not np.isfinite(self.theta):
            raise ValueError("theta must be finite.")

        self._validate_theta()

    def _validate_theta(self) -> None:
        """
        Validate family-specific restrictions on ``theta``.

        Subclasses may override this method.
        """

    @property
    def is_independence(self) -> bool:
        """Return whether the current parameter represents independence."""
        return self.theta == self.independence_parameter

    @final
    def cdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """Evaluate the copula cumulative distribution function."""
        u_array, v_array = self._prepare_inputs(u, v)

        if self.is_independence:
            return u_array * v_array

        return self._cdf(u_array, v_array)

    @final
    def pdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """Evaluate the copula probability density function."""
        u_array, v_array = self._prepare_inputs(u, v)

        if self.is_independence:
            return np.ones_like(u_array, dtype=np.float64)

        return self._pdf(u_array, v_array)

    @final
    def logpdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """Evaluate the logarithm of the copula density."""
        u_array, v_array = self._prepare_inputs(u, v)

        if self.is_independence:
            return np.zeros_like(u_array, dtype=np.float64)

        return self._logpdf(u_array, v_array)

    @abstractmethod
    def _cdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the family-specific CDF outside independence."""
        raise NotImplementedError

    @abstractmethod
    def _pdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the family-specific PDF outside independence."""
        raise NotImplementedError

    def _logpdf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Evaluate the family-specific log-density.

        Subclasses may override this method with an analytical expression for
        improved numerical stability.
        """
        with np.errstate(divide="ignore"):
            return np.log(self._pdf(u, v))
