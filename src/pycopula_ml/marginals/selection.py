"""
Automatic marginal distribution selection.

This module fits several candidate continuous distributions to a sample,
compares their diagnostics, and selects one model according to a configured
criterion.

The default criterion is AIC. The selected model is not interpreted as the
true underlying distribution; it is only the preferred model among the
candidate set under the chosen criterion.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import (
    expon,
    gamma,
    lognorm,
    norm,
    weibull_min,
)

from .base import MarginalModel
from .parametric import ParametricMarginal
from .result import MarginalFitResult


@dataclass
class MarginalCandidate:
    """
    Define a candidate distribution for marginal model selection.

    Parameters
    ----------
    name : str
        Human-readable model name.
    distribution : scipy.stats distribution
        Continuous SciPy probability distribution.
    fit_kwargs : dict, optional
        Keyword arguments passed to the distribution ``fit`` method.

    Examples
    --------
    A Gamma distribution with location fixed at zero can be defined as

    >>> from scipy.stats import gamma
    >>> candidate = MarginalCandidate(
    ...     name="gamma",
    ...     distribution=gamma,
    ...     fit_kwargs={"floc": 0.0},
    ... )
    """

    name: str
    distribution: Any
    fit_kwargs: Dict[str, Any] = field(
        default_factory=dict
    )


DEFAULT_CANDIDATES = (
    MarginalCandidate(
        "normal",
        norm,
    ),
    MarginalCandidate(
        "gamma",
        gamma,
    ),
    MarginalCandidate(
        "lognormal",
        lognorm,
    ),
    MarginalCandidate(
        "weibull",
        weibull_min,
    ),
    MarginalCandidate(
        "exponential",
        expon,
    ),
)


class MarginalSelector:
    """
    Fit, compare, and select parametric marginal distributions.

    Parameters
    ----------
    candidates : sequence of MarginalCandidate, optional
        Candidate distributions. If omitted, a default collection containing
        Normal, Gamma, Lognormal, Weibull, and Exponential distributions is
        used.
    criterion : {"aic", "bic", "ks"}, default="aic"
        Criterion minimized to select a model. ``ks`` refers to the
        Kolmogorov-Smirnov statistic, not its p-value.
    small_sample_warning : int, default=20
        Emit a diagnostic warning when the fitted sample contains fewer than
        this number of observations. This is a heuristic warning, not a
        formal statistical threshold.
    boundary_tolerance : float, default=1e-6
        Relative tolerance used to identify fitted support boundaries that
        are extremely close to the minimum or maximum observed value.

    Attributes
    ----------
    selected_model_ : ParametricMarginal or None
        Model selected according to ``criterion``.
    selected_result_ : MarginalFitResult or None
        Diagnostics for the selected model.
    results_ : list of MarginalFitResult
        Successful fits ordered by the configured criterion.
    failed_ : dict
        Candidate names mapped to fitting error messages.
    warnings_ : list of str
        Diagnostic warnings generated during selection.
    """

    _CRITERION_ATTRIBUTES = {
        "aic": "aic",
        "bic": "bic",
        "ks": "ks_statistic",
    }

    def __init__(
        self,
        candidates: Optional[
            Sequence[MarginalCandidate]
        ] = None,
        criterion: str = "aic",
        small_sample_warning: int = 20,
        boundary_tolerance: float = 1e-6,
    ) -> None:
        if criterion not in self._CRITERION_ATTRIBUTES:
            raise ValueError(
                "criterion must be 'aic', 'bic', or 'ks'."
            )

        if small_sample_warning < 1:
            raise ValueError(
                "small_sample_warning must be at least 1."
            )

        if boundary_tolerance <= 0.0:
            raise ValueError(
                "boundary_tolerance must be greater than zero."
            )

        self.candidates = list(
            candidates
            if candidates is not None
            else DEFAULT_CANDIDATES
        )

        self.criterion = criterion
        self.small_sample_warning = small_sample_warning
        self.boundary_tolerance = boundary_tolerance

        self.selected_model_: Optional[
            ParametricMarginal
        ] = None

        self.selected_result_: Optional[
            MarginalFitResult
        ] = None

        self.results_: List[
            MarginalFitResult
        ] = []

        self.failed_: Dict[
            str,
            str,
        ] = {}

        self.warnings_: List[str] = []

        self._fitted_models: List[
            ParametricMarginal
        ] = []

    def fit(
        self,
        values: ArrayLike,
    ) -> "MarginalSelector":
        """
        Fit all candidate distributions and select one model.

        Parameters
        ----------
        values : array_like
            One-dimensional numerical sample.

        Returns
        -------
        MarginalSelector
            Fitted selector.

        Raises
        ------
        RuntimeError
            If none of the candidate distributions can be fitted.
        """
        data = MarginalModel._prepare_sample(
            values
        )

        self.results_ = []
        self.failed_ = {}
        self.warnings_ = []
        self._fitted_models = []

        self.selected_model_ = None
        self.selected_result_ = None

        for candidate in self.candidates:
            model = ParametricMarginal(
                distribution=candidate.distribution,
                name=candidate.name,
                fit_kwargs=candidate.fit_kwargs,
            )

            try:
                model.fit(data)
            except Exception as exc:
                self.failed_[
                    candidate.name
                ] = str(exc)

                continue

            if model.result_ is None:
                self.failed_[
                    candidate.name
                ] = (
                    "The candidate did not produce a fit result."
                )

                continue

            self._fitted_models.append(
                model
            )

            self.results_.append(
                model.result_
            )

        if not self._fitted_models:
            raise RuntimeError(
                "No candidate marginal distribution "
                "could be fitted."
            )

        attribute = self._CRITERION_ATTRIBUTES[
            self.criterion
        ]

        self._fitted_models.sort(
            key=lambda model: getattr(
                model.result_,
                attribute,
            )
        )

        self.results_.sort(
            key=lambda result: getattr(
                result,
                attribute,
            )
        )

        self.selected_model_ = (
            self._fitted_models[0]
        )

        self.selected_result_ = (
            self.results_[0]
        )

        self._collect_diagnostics(
            data
        )

        return self

    def transform(
        self,
        values: ArrayLike,
        epsilon: float = 1e-10,
    ) -> NDArray[np.float64]:
        """
        Transform values using the selected marginal distribution.

        Parameters
        ----------
        values : array_like
            Values on the original measurement scale.
        epsilon : float, default=1e-10
            Clipping value used to avoid exact zero and one probabilities.

        Returns
        -------
        numpy.ndarray
            Probability-integral-transformed values.

        Raises
        ------
        RuntimeError
            If the selector has not been fitted.
        """
        model = self._require_selected_model()

        return model.transform(
            values,
            epsilon=epsilon,
        )

    def fit_transform(
        self,
        values: ArrayLike,
        epsilon: float = 1e-10,
    ) -> NDArray[np.float64]:
        """
        Fit candidate distributions and transform the same sample.

        Parameters
        ----------
        values : array_like
            One-dimensional numerical sample.
        epsilon : float, default=1e-10
            Clipping value used to avoid exact zero and one probabilities.

        Returns
        -------
        numpy.ndarray
            Transformed observations on the unit interval.
        """
        self.fit(values)

        return self.transform(
            values,
            epsilon=epsilon,
        )

    def ranking(
        self,
    ) -> List[MarginalFitResult]:
        """
        Return successful fits ordered by the selection criterion.

        Returns
        -------
        list of MarginalFitResult
            Model results from lowest to highest criterion value.

        Raises
        ------
        RuntimeError
            If the selector has not been fitted.
        """
        self._require_selected_model()

        return list(
            self.results_
        )

    def summary(self) -> str:
        """
        Return a human-readable marginal selection report.

        Returns
        -------
        str
            Report containing the ranking, selected model, criterion, and
            diagnostic warnings.

        Raises
        ------
        RuntimeError
            If the selector has not been fitted.
        """
        self._require_selected_model()

        if self.selected_result_ is None:
            raise RuntimeError(
                "The selector has not produced "
                "a selected result."
            )

        attribute = self._CRITERION_ATTRIBUTES[
            self.criterion
        ]

        minimum = getattr(
            self.selected_result_,
            attribute,
        )

        lines = [
            "Marginal distribution selection",
            "--------------------------------",
            (
                f"Criterion: "
                f"{self.criterion.upper()}"
            ),
            (
                f"Sample size: "
                f"{self.selected_result_.sample_size}"
            ),
            "",
            (
                f"{'Rank':>4}  "
                f"{'Distribution':<14} "
                f"{'AIC':>10} "
                f"{'BIC':>10} "
                f"{'KS':>10} "
                f"{'Delta':>10}"
            ),
        ]

        for rank, result in enumerate(
            self.results_,
            start=1,
        ):
            criterion_value = getattr(
                result,
                attribute,
            )

            delta = (
                criterion_value
                - minimum
            )

            lines.append(
                f"{rank:>4}  "
                f"{result.distribution:<14} "
                f"{result.aic:>10.4f} "
                f"{result.bic:>10.4f} "
                f"{result.ks_statistic:>10.4f} "
                f"{delta:>10.4f}"
            )

        lines.extend(
            [
                "",
                (
                    "Selected distribution: "
                    f"{self.selected_result_.distribution}"
                ),
                (
                    "Selection reason: lowest "
                    f"{self.criterion.upper()} "
                    "among successful candidates"
                ),
            ]
        )

        if self.failed_:
            lines.extend(
                [
                    "",
                    "Failed candidates:",
                ]
            )

            for name, message in self.failed_.items():
                lines.append(
                    f"- {name}: {message}"
                )

        if self.warnings_:
            lines.extend(
                [
                    "",
                    "Warnings:",
                ]
            )

            for message in self.warnings_:
                lines.append(
                    f"- {message}"
                )

        return "\n".join(
            lines
        )

    @property
    def best_model_(
        self,
    ) -> Optional[ParametricMarginal]:
        """
        Return the selected marginal model.

        Returns
        -------
        ParametricMarginal or None
            Alias for :attr:`selected_model_`.

        Notes
        -----
        ``selected_model_`` is the preferred name because the chosen model
        is only the best candidate under a specified criterion, not
        necessarily the true distribution.
        """
        return self.selected_model_

    @property
    def best_result_(
        self,
    ) -> Optional[MarginalFitResult]:
        """
        Return diagnostics for the selected marginal model.

        Returns
        -------
        MarginalFitResult or None
            Alias for :attr:`selected_result_`.
        """
        return self.selected_result_

    def _require_selected_model(
        self,
    ) -> ParametricMarginal:
        """
        Return the selected model or raise an error.

        Returns
        -------
        ParametricMarginal
            Selected fitted marginal model.

        Raises
        ------
        RuntimeError
            If the selector has not been fitted.
        """
        if self.selected_model_ is None:
            raise RuntimeError(
                "The selector must be fitted before use."
            )

        return self.selected_model_

    def _collect_diagnostics(
        self,
        data: NDArray[np.float64],
    ) -> None:
        """
        Collect warnings about potentially unstable model selection.

        Parameters
        ----------
        data : numpy.ndarray
            Sample used for fitting.
        """
        if data.size < self.small_sample_warning:
            self.warnings_.append(
                "The sample is small for automatic distribution "
                f"selection (n={data.size}). Review the selected "
                "model manually."
            )

        model = self._require_selected_model()

        if model.result_ is None:
            return

        if "floc" in model.fit_kwargs:
            return

        lower, upper = model.support()

        data_min = float(
            np.min(data)
        )

        data_max = float(
            np.max(data)
        )

        if (
            np.isfinite(lower)
            and self._is_close_to_boundary(
                lower,
                data_min,
            )
        ):
            self.warnings_.append(
                "The fitted lower support boundary is "
                "approximately equal to the minimum observed "
                "value. This may indicate a boundary-seeking "
                "or unstable fit."
            )

        if (
            np.isfinite(upper)
            and self._is_close_to_boundary(
                upper,
                data_max,
            )
        ):
            self.warnings_.append(
                "The fitted upper support boundary is "
                "approximately equal to the maximum observed "
                "value. This may indicate a boundary-seeking "
                "or unstable fit."
            )

    def _is_close_to_boundary(
        self,
        boundary: float,
        observed: float,
    ) -> bool:
        """
        Check whether a fitted support boundary is close to an observation.

        Parameters
        ----------
        boundary : float
            Fitted support boundary.
        observed : float
            Observed sample boundary.

        Returns
        -------
        bool
            True when the values are close under the configured tolerance.
        """
        tolerance = (
            self.boundary_tolerance
            * max(
                1.0,
                abs(observed),
            )
        )

        return (
            abs(boundary - observed)
            <= tolerance
        )
