"""Pairwise copula dependence matrices, corresponding to MLCOPULA build.weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pycopula_ml.estimation import fit_copula_mle

from .mutual_information import mutual_information


@dataclass(frozen=True)
class CopulaWeightMatrices:
    """
    Symmetric pairwise matrices for one fitted class or dataset.

    log_likelihood contains in-sample sums of log copula densities.
    theta contains each pair's fitted dependence parameter.
    mutual_information contains model-based MI in nats.
    weights is either log_likelihood or mutual_information.
    """

    log_likelihood: NDArray[np.float64]
    theta: NDArray[np.float64]
    mutual_information: NDArray[np.float64]
    weights: NDArray[np.float64]
    feature_names: tuple[str, ...]
    weight_method: str


def build_weight_matrices(
    U: ArrayLike,  # noqa: N803
    copula_class: type,
    *,
    bounds: tuple,
    weight_method: Literal["likelihood", "mutual_information"] = "mutual_information",
    feature_names: Sequence[str] | None = None,
    quadrature_order: int = 64,
) -> CopulaWeightMatrices:
    """
    Fit a bivariate copula for every pair of columns of ``U``.

    IMPORTANT: For supervised classification, call this separately for each
    class. To match MLCOPULA training, construct U as rank-based
    pseudo-observations within that class (pobs(X_class) in R). Marginal
    density fitting is a separate step for the eventual classifier.

    Parameters
    ----------
    U : array_like, shape (n_observations, n_features)
        Class-specific rank pseudo-observations in the open interval (0,1).
    copula_class : type
        Parametric bivariate copula class constructed by ``copula_class(theta=...)``.
    bounds : tuple
        Scalar MLE search interval or disjoint intervals understood by
        ``fit_copula_mle``. Specify suitable bounds for the chosen family.
    weight_method : {'likelihood', 'mutual_information'}
        Determines which computed matrix is returned as ``weights``.
    feature_names : sequence of str, optional
        Labels for columns; pandas DataFrame column names are inferred.
    quadrature_order : int, default 64
        Quadrature accuracy for non-Gaussian copulas.
    """
    if weight_method not in ("likelihood", "mutual_information"):
        raise ValueError("weight_method must be 'likelihood' or 'mutual_information'.")

    if feature_names is None and hasattr(U, "columns"):
        feature_names = list(U.columns)

    data = np.asarray(U, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError(
            "U must be 2D with at least two observations and two features."
        )
    if not np.all(np.isfinite(data)) or np.any((data <= 0.0) | (data >= 1.0)):
        raise ValueError(
            "Every entry of U must be finite and strictly between 0 and 1."
        )

    n_features = data.shape[1]
    if feature_names is None:
        names = tuple(f"X{i + 1}" for i in range(n_features))
    else:
        names = tuple(str(name) for name in feature_names)
        if len(names) != n_features or len(set(names)) != n_features:
            raise ValueError("feature_names must be unique and match U columns.")

    log_likelihood = np.zeros((n_features, n_features), dtype=float)
    theta = np.zeros_like(log_likelihood)
    mi = np.zeros_like(log_likelihood)

    for i, j in combinations(range(n_features), 2):
        fit = fit_copula_mle(copula_class, data[:, i], data[:, j], bounds=bounds)
        if not fit.success or not np.isfinite(fit.log_likelihood):
            raise RuntimeError(f"Copula MLE failed for {names[i]} and {names[j]}.")
        copula = copula_class(theta=float(fit.theta))
        dependence = mutual_information(copula, order=quadrature_order)

        log_likelihood[i, j] = log_likelihood[j, i] = float(fit.log_likelihood)
        theta[i, j] = theta[j, i] = float(fit.theta)
        mi[i, j] = mi[j, i] = dependence

    weights = mi if weight_method == "mutual_information" else log_likelihood
    return CopulaWeightMatrices(
        log_likelihood=log_likelihood,
        theta=theta,
        mutual_information=mi,
        weights=weights.copy(),
        feature_names=names,
        weight_method=weight_method,
    )
