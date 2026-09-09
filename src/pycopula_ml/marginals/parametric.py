"""
Parametric marginal probability models.

This module provides a generic wrapper around continuous probability
distributions implemented in :mod:`scipy.stats`.

SciPy performs parameter estimation and evaluates the PDF, log-PDF, and CDF.
The :class:`ParametricMarginal` class adds the interface and diagnostics
required by pycopula-ml.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import kstest

from .base import ContinuousMarginalModel
from .result import MarginalFitResult


class ParametricMarginal(ContinuousMarginalModel):
    """
    Parametric marginal model backed by a SciPy distribution.

    Parameters
    ----------
    distribution : scipy.stats distribution
        Continuous SciPy distribution implementing ``fit``, ``pdf``,
        ``logpdf``, ``cdf``, and ``support``.
    name : str, optional
        Human-readable name assigned to the fitted model. If omitted, the
        SciPy distribution name is used.
    fit_kwargs : dict, optional
        Keyword arguments passed directly to ``distribution.fit``. These can
        be used to fix parameters, for example ``{"floc": 0.0}``.

    Attributes
    ----------
    parameters_ : tuple of float or None
        Estimated distribution parameters after fitting.
    result_ : MarginalFitResult or None
        Statistical diagnostics for the fitted model.
    """

    def __init__(
        self,
        distribution: Any,
        name: Optional[str] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.distribution = distribution
        self.name = name or distribution.name
        self.fit_kwargs = dict(fit_kwargs or {})

        self.parameters_: Optional[Tuple[float, ...]] = None
        self.result_: Optional[MarginalFitResult] = None

    def fit(
        self,
        values: ArrayLike,
    ) -> "ParametricMarginal":
        """
        Estimate distribution parameters from observed data.

        Parameters
        ----------
        values : array_like
            One-dimensional numerical sample.

        Returns
        -------
        ParametricMarginal
            Fitted marginal model.

        Raises
        ------
        RuntimeError
            If fitting produces non-finite parameters or a non-finite
            log-likelihood.
        """
        data = self._prepare_sample(values)

        parameters = self.distribution.fit(
            data,
            **self.fit_kwargs,
        )

        self.parameters_ = tuple(
            float(parameter)
            for parameter in parameters
        )

        if not np.all(np.isfinite(self.parameters_)):
            self.parameters_ = None

            raise RuntimeError(
                f"{self.name} produced non-finite fitted parameters."
            )

        log_density = self.distribution.logpdf(
            data,
            *self.parameters_,
        )

        if not np.all(np.isfinite(log_density)):
            self.parameters_ = None

            raise RuntimeError(
                f"{self.name} produced a non-finite log-likelihood."
            )

        log_likelihood = float(
            np.sum(log_density)
        )

        n_parameters = self._count_free_parameters()
        sample_size = int(data.size)

        aic = (
            2.0 * n_parameters
            - 2.0 * log_likelihood
        )

        bic = (
            np.log(sample_size) * n_parameters
            - 2.0 * log_likelihood
        )

        ks_result = kstest(
            data,
            self.distribution.cdf,
            args=self.parameters_,
        )

        self.result_ = MarginalFitResult(
            distribution=self.name,
            parameters=self.parameters_,
            sample_size=sample_size,
            n_parameters=n_parameters,
            log_likelihood=log_likelihood,
            aic=float(aic),
            bic=float(bic),
            ks_statistic=float(ks_result.statistic),
            ks_pvalue=float(ks_result.pvalue),
        )

        return self

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
            Fitted probability density values.
        """
        parameters = self._require_fitted()
        data = self._prepare_evaluation_values(values)

        return np.asarray(
            self.distribution.pdf(
                data,
                *parameters,
            ),
            dtype=np.float64,
        )

    def logpdf(
        self,
        values: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate the fitted log-probability density function.

        Parameters
        ----------
        values : array_like
            Values at which the log-density is evaluated.

        Returns
        -------
        numpy.ndarray
            Natural logarithm of the fitted density.
        """
        parameters = self._require_fitted()
        data = self._prepare_evaluation_values(values)

        return np.asarray(
            self.distribution.logpdf(
                data,
                *parameters,
            ),
            dtype=np.float64,
        )

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
            Fitted cumulative probabilities.
        """
        parameters = self._require_fitted()
        data = self._prepare_evaluation_values(values)

        return np.asarray(
            self.distribution.cdf(
                data,
                *parameters,
            ),
            dtype=np.float64,
        )

    def support(self) -> Tuple[float, float]:
        """
        Return the support of the fitted distribution.

        Returns
        -------
        tuple of float
            Lower and upper limits of the fitted support.
        """
        parameters = self._require_fitted()

        shape_parameters, loc, scale = self._split_parameters(
            parameters
        )

        lower, upper = self.distribution.support(
            *shape_parameters,
            loc=loc,
            scale=scale,
        )

        return float(lower), float(upper)

    @property
    def loc_(self) -> float:
        """
        Return the fitted location parameter.

        Returns
        -------
        float
            Fitted ``loc`` parameter.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        parameters = self._require_fitted()

        _, loc, _ = self._split_parameters(
            parameters
        )

        return loc

    @property
    def scale_(self) -> float:
        """
        Return the fitted scale parameter.

        Returns
        -------
        float
            Fitted ``scale`` parameter.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        parameters = self._require_fitted()

        _, _, scale = self._split_parameters(
            parameters
        )

        return scale

    def _require_fitted(self) -> Tuple[float, ...]:
        """
        Return fitted parameters or raise an error.

        Returns
        -------
        tuple of float
            Estimated distribution parameters.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self.parameters_ is None:
            raise RuntimeError(
                "The marginal model must be fitted before use."
            )

        return self.parameters_

    def _shape_names(self) -> Tuple[str, ...]:
        """
        Return SciPy shape-parameter names for the distribution.

        Returns
        -------
        tuple of str
            Shape-parameter names in SciPy order.
        """
        shapes = getattr(
            self.distribution,
            "shapes",
            None,
        )

        if not shapes:
            return ()

        return tuple(
            name.strip()
            for name in shapes.split(",")
        )

    def _split_parameters(
        self,
        parameters: Tuple[float, ...],
    ) -> Tuple[Tuple[float, ...], float, float]:
        """
        Split SciPy parameters into shape, location, and scale components.

        Parameters
        ----------
        parameters : tuple of float
            Parameters returned by a SciPy continuous distribution fit.

        Returns
        -------
        tuple
            ``(shape_parameters, loc, scale)``.
        """
        n_shapes = len(
            self._shape_names()
        )

        shape_parameters = parameters[:n_shapes]
        loc = float(
            parameters[n_shapes]
        )
        scale = float(
            parameters[n_shapes + 1]
        )

        return shape_parameters, loc, scale

    def _count_free_parameters(self) -> int:
        """
        Count parameters estimated freely from the sample.

        Returns
        -------
        int
            Number of non-fixed fitted parameters.

        Notes
        -----
        SciPy allows shape, location, and scale parameters to be fixed using
        arguments such as ``f0``, ``f<shape_name>``, ``floc``, and
        ``fscale``. Fixed parameters are excluded from the AIC and BIC
        parameter count.
        """
        parameters = self._require_fitted()
        shape_names = self._shape_names()

        fixed_parameters = set()

        for index, shape_name in enumerate(
            shape_names
        ):
            accepted_keys = {
                f"f{index}",
                f"f{shape_name}",
                f"fix_{shape_name}",
            }

            if any(
                key in self.fit_kwargs
                for key in accepted_keys
            ):
                fixed_parameters.add(
                    ("shape", index)
                )

        if "floc" in self.fit_kwargs:
            fixed_parameters.add(
                ("loc", 0)
            )

        if "fscale" in self.fit_kwargs:
            fixed_parameters.add(
                ("scale", 0)
            )

        n_free = (
            len(parameters)
            - len(fixed_parameters)
        )

        if n_free <= 0:
            raise RuntimeError(
                "The fitted model must contain at least one "
                "free parameter."
            )

        return n_free
