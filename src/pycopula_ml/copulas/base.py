"""
Base classes for bivariate copula models.

This module defines the common interface for continuous bivariate copulas
implemented in :mod:`pycopula_lm.copulas`.

A continuous bivariate copula is a cumulative distribution function defined
on the unit square [0, 1]^2. Its probability density function is obtained
from the mixed second derivative of the copula CDF.

The :class:`BivariateCopula` abstract base class establishes the minimum
interface required by the library:

- cumulative distribution function (CDF),
- probability density function (PDF),
- log-probability density function (log-PDF),
- sample log-likelihood.

Concrete copula families, such as Frank, Clayton, or Gumbel, must implement
their own CDF and PDF. They may also override the default log-PDF
implementation when a numerically more stable analytical expression is
available.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BivariateCopula(ABC):
    """
    Abstract base class for continuous bivariate copulas.

    A bivariate copula models the dependence structure between two random
    variables after their marginal distributions have been transformed to
    uniform random variables ``U`` and ``V`` on the interval [0, 1].

    Concrete subclasses must implement :meth:`cdf` and :meth:`pdf`.

    The base class provides generic implementations of :meth:`logpdf` and
    :meth:`log_likelihood`, since their definitions are common to continuous
    bivariate copulas.
    """

    @abstractmethod
    def cdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the copula cumulative distribution function.

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
            Values of the copula cumulative distribution function evaluated
            at the points ``(u, v)``.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the copula probability density function.

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
            Values of the copula probability density function evaluated at
            the points ``(u, v)``.
        """
        raise NotImplementedError

    def logpdf(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the logarithm of the copula density.

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
            Natural logarithm of the copula density evaluated at ``(u, v)``.
        """
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(u, v))

    def log_likelihood(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> float:
        """
        Compute the copula log-likelihood for a sample.

        Parameters
        ----------
        u : array_like
            Observations of the first uniform marginal variable.
        v : array_like
            Observations of the second uniform marginal variable.

        Returns
        -------
        float
            Sum of the log-density values over all observations.
        """
        return float(np.sum(self.logpdf(u, v)))

    @staticmethod
    def _prepare_inputs(
        u: ArrayLike,
        v: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Convert, broadcast, and validate copula input values.

        Parameters
        ----------
        u : array_like
            Values of the first uniform marginal variable.
        v : array_like
            Values of the second uniform marginal variable.

        Returns
        -------
        tuple of numpy.ndarray
            Two broadcast-compatible arrays containing the validated values
            of ``u`` and ``v``.

        Raises
        ------
        ValueError
            If ``u`` and ``v`` cannot be broadcast to a common shape.
        ValueError
            If either input contains non-finite values.
        ValueError
            If any value lies outside the interval [0, 1].
        """
        u_array = np.asarray(u, dtype=np.float64)
        v_array = np.asarray(v, dtype=np.float64)

        try:
            u_array, v_array = np.broadcast_arrays(u_array, v_array)
        except ValueError as exc:
            raise ValueError(
                "u and v must be broadcast-compatible."
            ) from exc

        if not np.all(np.isfinite(u_array)):
            raise ValueError("u must contain only finite values.")

        if not np.all(np.isfinite(v_array)):
            raise ValueError("v must contain only finite values.")

        if np.any((u_array < 0.0) | (u_array > 1.0)):
            raise ValueError(
                "u must contain values in the interval [0, 1]."
            )

        if np.any((v_array < 0.0) | (v_array > 1.0)):
            raise ValueError(
                "v must contain values in the interval [0, 1]."
            )

        return u_array, v_array
