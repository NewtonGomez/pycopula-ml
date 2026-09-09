"""
Marginal distribution modeling and selection.

This package provides tools for fitting univariate probability
distributions, comparing candidate marginal models, estimating empirical
marginals, and transforming observations to the unit interval for
copula-based modeling.

The public API includes parametric and empirical approaches.
"""

from .base import (
    ContinuousMarginalModel,
    MarginalModel,
)
from .empirical import (
    EmpiricalMarginal,
    bivariate_pseudo_observations,
    pseudo_observations,
)
from .parametric import ParametricMarginal
from .result import MarginalFitResult
from .selection import (
    DEFAULT_CANDIDATES,
    MarginalCandidate,
    MarginalSelector,
)

__all__ = [
    "ContinuousMarginalModel",
    "DEFAULT_CANDIDATES",
    "EmpiricalMarginal",
    "MarginalCandidate",
    "MarginalFitResult",
    "MarginalModel",
    "MarginalSelector",
    "ParametricMarginal",
    "bivariate_pseudo_observations",
    "pseudo_observations",
]
