"""Maximum-likelihood estimation utilities for parametric copulas.

This module provides a generic scalar optimizer for bivariate copula models
whose dependence structure is controlled by a single parameter named
``theta``.

The optimizer expects pseudo-observations in the open unit interval ``(0, 1)``
and maximizes the copula log-likelihood by minimizing its negative value.

Examples
--------
Estimate the parameter of a Frank copula over its positive and negative
parameter regions while excluding the independence limit ``theta = 0``::

    result = fit_copula_mle(
        FrankCopula,
        u,
        v,
        bounds=((-50.0, -1e-6), (1e-6, 50.0)),
    )

    print(result.theta)
    print(result.log_likelihood)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, minimize_scalar


Bounds = Tuple[float, float]
BoundsInput = Union[Bounds, Sequence[Bounds]]


class CopulaLikelihood(Protocol):
    """Protocol required from a fitted copula instance."""

    def log_likelihood(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> float:
        """Return the sample copula log-likelihood."""


class CopulaFactory(Protocol):
    """Protocol for a copula class or factory parameterized by ``theta``."""

    def __call__(self, *, theta: float) -> CopulaLikelihood:
        """Create a copula instance with the requested dependence parameter."""


@dataclass(frozen=True)
class CopulaFitResult:
    """Result of one-dimensional maximum-likelihood copula fitting.

    Attributes
    ----------
    theta : float
        Maximum-likelihood estimate of the dependence parameter.
    log_likelihood : float
        Maximized copula log-likelihood.
    success : bool
        Whether the selected SciPy optimization converged successfully.
    message : str
        Convergence message returned by SciPy.
    nfev : int
        Number of objective-function evaluations for the selected interval.
    n_obs : int
        Number of paired pseudo-observations used for estimation.
    interval : tuple[float, float]
        Parameter interval that produced the selected estimate.
    raw_result : scipy.optimize.OptimizeResult
        Original SciPy result for the selected interval.
    interval_results : tuple[scipy.optimize.OptimizeResult, ...]
        Results for all intervals that were searched.
    """

    theta: float
    log_likelihood: float
    success: bool
    message: str
    nfev: int
    n_obs: int
    interval: Bounds
    raw_result: OptimizeResult
    interval_results: Tuple[OptimizeResult, ...]


def _prepare_pseudo_observations(
    u: ArrayLike,
    v: ArrayLike,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and convert paired pseudo-observations.

    Pseudo-observations used in a copula likelihood must be finite, paired,
    one-dimensional, and strictly inside the unit interval.
    """
    u_array = np.asarray(u, dtype=float)
    v_array = np.asarray(v, dtype=float)

    if u_array.ndim != 1 or v_array.ndim != 1:
        raise ValueError("u and v must be one-dimensional arrays.")

    if u_array.size == 0 or v_array.size == 0:
        raise ValueError("u and v must not be empty.")

    if u_array.shape != v_array.shape:
        raise ValueError("u and v must have the same shape.")

    if not np.all(np.isfinite(u_array)) or not np.all(np.isfinite(v_array)):
        raise ValueError("u and v must contain only finite values.")

    if np.any((u_array <= 0.0) | (u_array >= 1.0)):
        raise ValueError("All values in u must lie strictly inside (0, 1).")

    if np.any((v_array <= 0.0) | (v_array >= 1.0)):
        raise ValueError("All values in v must lie strictly inside (0, 1).")

    return u_array, v_array


def _normalize_bounds(bounds: BoundsInput) -> Tuple[Bounds, ...]:
    """Normalize one interval or several intervals into a tuple of bounds."""
    if len(bounds) == 2 and all(np.isscalar(value) for value in bounds):
        raw_intervals = (bounds,)
    else:
        raw_intervals = tuple(bounds)

    if not raw_intervals:
        raise ValueError("At least one parameter interval is required.")

    normalized = []

    for interval in raw_intervals:
        if len(interval) != 2:
            raise ValueError(
                "Each parameter interval must contain exactly two values."
            )

        lower = float(interval[0])
        upper = float(interval[1])

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("Parameter bounds must be finite.")

        if lower >= upper:
            raise ValueError(
                "The lower parameter bound must be smaller than the upper bound."
            )

        normalized.append((lower, upper))

    return tuple(normalized)


def _negative_log_likelihood(
    theta: float,
    copula: CopulaFactory,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> float:
    """Return the negative copula log-likelihood for a candidate parameter."""
    model = copula(theta=float(theta))
    log_likelihood = float(model.log_likelihood(u, v))

    if not np.isfinite(log_likelihood):
        return np.inf

    return -log_likelihood


def fit_copula_mle(
    copula: CopulaFactory,
    u: ArrayLike,
    v: ArrayLike,
    bounds: BoundsInput,
    *,
    xatol: float = 1e-8,
    maxiter: int = 500,
) -> CopulaFitResult:
    """Estimate a scalar copula parameter by maximum likelihood.

    Parameters
    ----------
    copula : CopulaFactory
        Copula class or factory accepting ``theta`` as a keyword argument and
        returning an object with a ``log_likelihood(u, v)`` method.
    u, v : array-like
        Paired pseudo-observations. Values must lie strictly inside ``(0, 1)``.
    bounds : tuple[float, float] or sequence of tuple[float, float]
        One parameter interval or several disjoint intervals to search.
        Multiple intervals are useful when a copula has an excluded parameter
        value, such as ``theta = 0`` for the Frank family.
    xatol : float, default=1e-8
        Absolute tolerance in the estimated parameter used by SciPy's bounded
        scalar optimizer.
    maxiter : int, default=500
        Maximum number of optimizer iterations per interval.

    Returns
    -------
    CopulaFitResult
        Maximum-likelihood estimate and optimization diagnostics.

    Raises
    ------
    ValueError
        If pseudo-observations or bounds are invalid.
    RuntimeError
        If no searched interval produces a finite successful optimization.
    """
    u_array, v_array = _prepare_pseudo_observations(u, v)
    intervals = _normalize_bounds(bounds)

    if xatol <= 0.0 or not np.isfinite(xatol):
        raise ValueError("xatol must be a finite positive number.")

    if maxiter <= 0:
        raise ValueError("maxiter must be a positive integer.")

    results = []

    for interval in intervals:
        result = minimize_scalar(
            _negative_log_likelihood,
            args=(copula, u_array, v_array),
            bounds=interval,
            method="bounded",
            options={
                "xatol": xatol,
                "maxiter": maxiter,
            },
        )
        result["searched_interval"] = interval
        results.append(result)

    valid_results = [
        result
        for result in results
        if bool(result.success) and np.isfinite(result.fun)
    ]

    if not valid_results:
        messages = "; ".join(str(result.message) for result in results)
        raise RuntimeError(
            "Copula parameter optimization failed for every interval. "
            f"SciPy messages: {messages}"
        )

    best = min(valid_results, key=lambda result: float(result.fun))
    best_interval = tuple(best["searched_interval"])

    return CopulaFitResult(
        theta=float(best.x),
        log_likelihood=-float(best.fun),
        success=bool(best.success),
        message=str(best.message),
        nfev=int(best.nfev),
        n_obs=int(u_array.size),
        interval=(float(best_interval[0]), float(best_interval[1])),
        raw_result=best,
        interval_results=tuple(results),
    )
