"""
Base classes for marginal probability models.

This module defines the common interfaces used by marginal distribution
models in :mod:`pycopula_ml.marginals`.

All marginal models provide a cumulative distribution function and a
transformation from observations on their original scale to values on the
unit interval.

Continuous marginal models additionally provide probability density and
log-density functions.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


class MarginalModel(ABC):
    """
    Abstract base class for univariate marginal probability models.

    A marginal model represents the distribution of a single random
    variable independently of its dependence structure with other
    variables.

    Once fitted, its cumulative distribution function can be used to
    transform observations according to

    .. math::

        U = F_X(X),

    which produces values on the unit interval suitable for copula
    modeling.
    """

    @abstractmethod
    def fit(
        self,
        values: ArrayLike,
    ) -> "MarginalModel":
        """
        Fit the marginal model to observed data.

        Parameters
        ----------
        values : array_like
            One-dimensional numerical sample.

        Returns
        -------
        MarginalModel
            Fitted marginal model.
        """
        raise NotImplementedError

    @abstractmethod
    def cdf(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the fitted cumulative distribution function.

        Parameters
        ----------
        values : array_like
            Values at which the CDF is evaluated.

        Returns
        -------
        numpy.ndarray
            Cumulative probabilities.
        """
        raise NotImplementedError

    def transform(
        self,
        values: ArrayLike,
        epsilon: float = 1e-10,
    ) -> NDArray[np.float64]:
        """
        Transform observations to the open unit interval.

        The transformation is based on the fitted marginal CDF:

        .. math::

            u_i = F_X(x_i).

        Exact zero and one values are clipped because copula densities and
        log-densities may be numerically problematic at the boundaries.

        Parameters
        ----------
        values : array_like
            Values to transform.
        epsilon : float, default=1e-10
            Clipping value used to constrain transformed values to
            ``[epsilon, 1 - epsilon]``.

        Returns
        -------
        numpy.ndarray
            Transformed observations on the open unit interval.

        Raises
        ------
        ValueError
            If ``epsilon`` is not strictly between zero and 0.5.
        """
        if not 0.0 < epsilon < 0.5:
            raise ValueError(
                "epsilon must be strictly between 0 and 0.5."
            )

        probabilities = self.cdf(values)

        return np.clip(
            probabilities,
            epsilon,
            1.0 - epsilon,
        )

    @staticmethod
    def _prepare_sample(
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Convert and validate a sample used for model fitting.

        Parameters
        ----------
        values : array_like
            Input observations.

        Returns
        -------
        numpy.ndarray
            One-dimensional floating-point sample.

        Raises
        ------
        ValueError
            If the sample is not one-dimensional.
        ValueError
            If the sample is empty.
        ValueError
            If the sample contains non-finite values.
        """
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.ndim != 1:
            raise ValueError(
                "values must be one-dimensional."
            )

        if array.size == 0:
            raise ValueError(
                "values must contain at least one observation."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "values must contain only finite observations."
            )

        return array

    @staticmethod
    def _prepare_evaluation_values(
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Convert and validate values used for probability evaluation.

        Parameters
        ----------
        values : array_like
            Values at which a probability function is evaluated.

        Returns
        -------
        numpy.ndarray
            Floating-point representation of the supplied values.

        Raises
        ------
        ValueError
            If any value is non-finite.
        """
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "values must contain only finite observations."
            )

        return array


class ContinuousMarginalModel(MarginalModel):
    """
    Abstract base class for continuous marginal probability models.

    Continuous marginal distributions provide probability density and
    log-density functions in addition to the common marginal CDF.
    """

    @abstractmethod
    def pdf(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the fitted probability density function.

        Parameters
        ----------
        values : array_like
            Values at which the density is evaluated.

        Returns
        -------
        numpy.ndarray
            Probability density values.
        """
        raise NotImplementedError

    @abstractmethod
    def logpdf(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the logarithm of the fitted probability density.

        Parameters
        ----------
        values : array_like
            Values at which the log-density is evaluated.

        Returns
        -------
        numpy.ndarray
            Natural logarithm of the density values.
        """
        raise NotImplementedError
