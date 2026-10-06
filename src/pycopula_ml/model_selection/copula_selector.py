"""Selection utilities for fitted bivariate parametric copulas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..copulas.base import BivariateCopula
from ..estimation.maximum_likelihood import fit_copula_mle
from .criteria import aic, bic


Bounds: TypeAlias = tuple[float, float]
CopulaType: TypeAlias = type[BivariateCopula]
SelectionCriterion: TypeAlias = Literal["log_likelihood", "aic", "bic"]


@dataclass(frozen=True, slots=True)
class CopulaCriteria:
    """Fit statistics for one candidate copula.

    Attributes
    ----------
    copula:
        Copula class evaluated by the selector.
    theta:
        Maximum-likelihood estimate of the dependence parameter.
    log_likelihood:
        Maximized copula log-likelihood.
    aic:
        Akaike Information Criterion.
    bic:
        Bayesian Information Criterion.
    """

    copula: CopulaType
    theta: float
    log_likelihood: float
    aic: float
    bic: float

    @property
    def name(self) -> str:
        """Return the candidate copula class name."""
        return self.copula.__name__


@dataclass(frozen=True, slots=True)
class CopulaSelectionResult:
    """Result returned by :func:`copula_selector`.

    Attributes
    ----------
    selected:
        Candidate preferred according to ``criterion``.
    candidates:
        All fitted candidates, ordered from best to worst according to the
        selected criterion.
    criterion:
        Criterion used for model selection.
    """

    selected: CopulaCriteria
    candidates: tuple[CopulaCriteria, ...]
    criterion: SelectionCriterion


def _prepare_inputs(
    u: ArrayLike,
    v: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and convert bivariate pseudo-observations."""
    u_array = np.asarray(u, dtype=float)
    v_array = np.asarray(v, dtype=float)

    if u_array.ndim != 1 or v_array.ndim != 1:
        raise ValueError("u and v must be one-dimensional arrays.")

    if u_array.size != v_array.size:
        raise ValueError("u and v must contain the same number of observations.")

    if u_array.size == 0:
        raise ValueError("u and v must contain at least one observation.")

    if not np.all(np.isfinite(u_array)) or not np.all(np.isfinite(v_array)):
        raise ValueError("u and v must contain only finite values.")

    return u_array, v_array


def _validate_candidate_copulas(
    candidate_copulas: Mapping[CopulaType, Bounds],
) -> None:
    """Validate the candidate-copula mapping."""
    if not candidate_copulas:
        raise ValueError("At least one candidate copula must be provided.")

    for copula, bounds in candidate_copulas.items():
        if not isinstance(copula, type) or not issubclass(copula, BivariateCopula):
            raise TypeError(
                "Each candidate key must be a BivariateCopula subclass."
            )

        if len(bounds) != 2:
            raise ValueError(
                f"Bounds for {copula.__name__} must contain exactly two values."
            )

        lower, upper = bounds

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(
                f"Bounds for {copula.__name__} must be finite."
            )

        if lower >= upper:
            raise ValueError(
                f"Lower bound must be smaller than upper bound for "
                f"{copula.__name__}."
            )


def _sort_candidates(
    candidates: list[CopulaCriteria],
    criterion: SelectionCriterion,
) -> list[CopulaCriteria]:
    """Sort fitted candidates from best to worst."""
    if criterion == "log_likelihood":
        return sorted(
            candidates,
            key=lambda candidate: candidate.log_likelihood,
            reverse=True,
        )

    if criterion == "aic":
        return sorted(candidates, key=lambda candidate: candidate.aic)

    if criterion == "bic":
        return sorted(candidates, key=lambda candidate: candidate.bic)

    raise ValueError(
        "criterion must be one of: 'log_likelihood', 'aic', or 'bic'."
    )


def _print_results(
    candidates: tuple[CopulaCriteria, ...],
    criterion: SelectionCriterion,
) -> None:
    """Print a compact summary table of the fitted candidate copulas."""
    print("Copula model selection")
    print("-" * 72)
    print(
        f"{'Copula':<24}"
        f"{'theta':>11}"
        f"{'logLik':>13}"
        f"{'AIC':>12}"
        f"{'BIC':>12}"
    )
    print("-" * 72)

    for candidate in candidates:
        print(
            f"{candidate.name:<24}"
            f"{candidate.theta:>11.6f}"
            f"{candidate.log_likelihood:>13.6f}"
            f"{candidate.aic:>12.6f}"
            f"{candidate.bic:>12.6f}"
        )

    print("-" * 72)
    print(f"Criterion: {criterion}")
    print(f"Selected copula: {candidates[0].name}")


def copula_selector(
    candidate_copulas: Mapping[CopulaType, Bounds],
    u: ArrayLike,
    v: ArrayLike,
    *,
    criterion: SelectionCriterion = "bic",
    n_parameters: int = 1,
    print_results: bool = False,
) -> CopulaSelectionResult:
    """Fit and compare candidate parametric bivariate copulas.

    Each candidate copula is fitted by maximum likelihood. The fitted models
    are then compared using the maximized log-likelihood, AIC, or BIC.

    Parameters
    ----------
    candidate_copulas:
        Mapping whose keys are ``BivariateCopula`` subclasses and whose values
        are ``(lower, upper)`` bounds used during maximum-likelihood
        estimation of the dependence parameter.
    u, v:
        One-dimensional pseudo-observation arrays. One statistical observation
        is the pair ``(u[i], v[i])``.
    criterion:
        Model-selection criterion. ``"log_likelihood"`` selects the largest
        value, while ``"aic"`` and ``"bic"`` select the smallest value.
    n_parameters:
        Number of free copula parameters. The current parametric copulas in
        ``pycopula_ml`` use one dependence parameter, so the default is 1.
    print_results:
        If ``True``, print a formatted comparison table.

    Returns
    -------
    CopulaSelectionResult
        Selected candidate together with all fitted candidates ordered from
        best to worst according to ``criterion``.

    Raises
    ------
    ValueError
        If the input vectors are invalid, the candidate mapping is empty, or
        the selection criterion is unknown.
    TypeError
        If a candidate is not a ``BivariateCopula`` subclass.

    Notes
    -----
    All candidate copulas must be fitted to the same ``u`` and ``v`` vectors
    for their likelihood-based criteria to be directly comparable.
    """
    u_array, v_array = _prepare_inputs(u, v)
    _validate_candidate_copulas(candidate_copulas)

    candidates: list[CopulaCriteria] = []

    for copula, bounds in candidate_copulas.items():
        fit_result = fit_copula_mle(
            copula,
            u_array,
            v_array,
            bounds=bounds,
        )

        candidates.append(
            CopulaCriteria(
                copula=copula,
                theta=float(fit_result.theta),
                log_likelihood=float(fit_result.log_likelihood),
                aic=aic(
                    fit_result.log_likelihood,
                    n_parameters,
                ),
                bic=bic(
                    fit_result.log_likelihood,
                    n_parameters,
                    u_array.size,
                ),
            )
        )

    ranked_candidates = tuple(
        _sort_candidates(candidates, criterion)
    )

    result = CopulaSelectionResult(
        selected=ranked_candidates[0],
        candidates=ranked_candidates,
        criterion=criterion,
    )

    if print_results:
        _print_results(
            result.candidates,
            result.criterion,
        )

    return result
