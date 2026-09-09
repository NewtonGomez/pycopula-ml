"""
Empirical marginal distribution utilities.

This module implements non-parametric marginal transformations for copula
modeling.

Two related but distinct concepts are provided:

1. :class:`EmpiricalMarginal`, which estimates a marginal cumulative
   distribution function using the empirical CDF.

2. :func:`pseudo_observations`, which transforms a sample using normalized
   ranks and is commonly used for copula estimation when no parametric
   marginal distribution is assumed.

The empirical CDF is defined as

.. math::

    \\widehat{F}_n(x)
    =
    \\frac{1}{n}
    \\sum_{i=1}^{n}
    \\mathbf{1}(X_i \\leq x).

Pseudo-observations are defined here as

.. math::

    u_i
    =
    \\frac{R_i}{n + 1},

where ``R_i`` is the rank of observation ``i``.
"""

from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

from .base import MarginalModel


class EmpiricalMarginal(MarginalModel):
    """
    Empirical marginal distribution based on the sample ECDF.

    The empirical cumulative distribution function is

    .. math::

        \\widehat{F}_n(x)
        =
        \\frac{
            \\#\\{X_i \\leq x\\}
        }{
            n
        }.

    Unlike a parametric marginal model, this class does not assume a
    theoretical probability distribution such as Normal, Gamma, or
    Lognormal.

    Attributes
    ----------
    sample_ : numpy.ndarray or None
        Original fitted sample.
    sorted_sample_ : numpy.ndarray or None
        Fitted sample sorted in ascending order.
    n_samples_ : int or None
        Number of observations used for fitting.
    """

    def __init__(self) -> None:
        self.sample_: Optional[
            NDArray[np.float64]
        ] = None

        self.sorted_sample_: Optional[
            NDArray[np.float64]
        ] = None

        self.n_samples_: Optional[int] = None

    def fit(
        self,
        values: ArrayLike,
    ) -> "EmpiricalMarginal":
        """
        Fit the empirical marginal distribution.

        Parameters
        ----------
        values : array_like
            One-dimensional numerical sample.

        Returns
        -------
        EmpiricalMarginal
            Fitted empirical marginal model.
        """
        data = self._prepare_sample(values)

        self.sample_ = data.copy()
        self.sorted_sample_ = np.sort(data)
        self.n_samples_ = int(data.size)

        return self

    def cdf(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the empirical cumulative distribution function.

        For each value ``x``, the ECDF is

        .. math::

            \\widehat{F}_n(x)
            =
            \\frac{
                \\#\\{X_i \\leq x\\}
            }{
                n
            }.

        Parameters
        ----------
        values : array_like
            Values at which the empirical CDF is evaluated.

        Returns
        -------
        numpy.ndarray
            Empirical cumulative probabilities.

        Raises
        ------
        RuntimeError
            If the empirical marginal has not been fitted.
        """
        sorted_sample = self._require_fitted()
        evaluation_values = self._prepare_evaluation_values(
            values
        )

        counts = np.searchsorted(
            sorted_sample,
            evaluation_values,
            side="right",
        )

        return np.asarray(
            counts / sorted_sample.size,
            dtype=np.float64,
        )

    def fit_transform(
        self,
        values: ArrayLike,
        epsilon: float = 1e-10,
    ) -> NDArray[np.float64]:
        """
        Fit the empirical marginal and transform the same sample.

        The transformation uses the fitted empirical CDF followed by
        clipping to the open unit interval.

        Parameters
        ----------
        values : array_like
            One-dimensional numerical sample.
        epsilon : float, default=1e-10
            Boundary clipping value.

        Returns
        -------
        numpy.ndarray
            ECDF-transformed observations.
        """
        self.fit(values)

        return self.transform(
            values,
            epsilon=epsilon,
        )

    def _require_fitted(
        self,
    ) -> NDArray[np.float64]:
        """
        Return the fitted sorted sample or raise an error.

        Returns
        -------
        numpy.ndarray
            Sorted fitted sample.

        Raises
        ------
        RuntimeError
            If the empirical marginal has not been fitted.
        """
        if self.sorted_sample_ is None:
            raise RuntimeError(
                "The empirical marginal must be fitted before use."
            )

        return self.sorted_sample_


def pseudo_observations(
    values: ArrayLike,
    ties: Literal[
        "average",
        "min",
        "max",
        "dense",
        "ordinal",
    ] = "average",
) -> NDArray[np.float64]:
    """
    Transform a sample into copula pseudo-observations.

    Pseudo-observations are calculated as

    .. math::

        u_i
        =
        \\frac{R_i}{n + 1},

    where ``R_i`` is the rank of observation ``i`` and ``n`` is the sample
    size.

    Parameters
    ----------
    values : array_like
        One-dimensional numerical sample.
    ties : {"average", "min", "max", "dense", "ordinal"},
        default="average"
        Ranking method used when tied values are present. The options follow
        :func:`scipy.stats.rankdata`.

    Returns
    -------
    numpy.ndarray
        Rank-based pseudo-observations in the open unit interval.

    Raises
    ------
    ValueError
        If the input sample is invalid.
    """
    data = MarginalModel._prepare_sample(
        values
    )

    ranks = rankdata(
        data,
        method=ties,
    )

    return np.asarray(
        ranks / (data.size + 1.0),
        dtype=np.float64,
    )


def bivariate_pseudo_observations(
    x: ArrayLike,
    y: ArrayLike,
    ties: Literal[
        "average",
        "min",
        "max",
        "dense",
        "ordinal",
    ] = "average",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Transform two paired samples into copula pseudo-observations.

    Parameters
    ----------
    x : array_like
        Observations of the first variable.
    y : array_like
        Observations of the second variable.
    ties : {"average", "min", "max", "dense", "ordinal"},
        default="average"
        Ranking method used when tied values are present.

    Returns
    -------
    tuple of numpy.ndarray
        Pseudo-observations ``(u, v)`` associated with ``x`` and ``y``.

    Raises
    ------
    ValueError
        If the two samples do not contain the same number of observations.
    """
    x_array = MarginalModel._prepare_sample(
        x
    )

    y_array = MarginalModel._prepare_sample(
        y
    )

    if x_array.size != y_array.size:
        raise ValueError(
            "x and y must contain the same number of observations."
        )

    u = pseudo_observations(
        x_array,
        ties=ties,
    )

    v = pseudo_observations(
        y_array,
        ties=ties,
    )

    return u, v
