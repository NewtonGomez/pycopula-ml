"""Model-selection tools for bivariate copula models."""

from .copula_selector import (
    CopulaCriteria,
    CopulaSelectionResult,
    copula_selector,
)
from .criteria import aic, bic

__all__ = [
    "CopulaCriteria",
    "CopulaSelectionResult",
    "aic",
    "bic",
    "copula_selector",
]
