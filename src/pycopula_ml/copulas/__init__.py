"""
Copula probability models.

This package contains the copula models implemented by pycopula-ml.

All continuous bivariate copulas derive from
:class:`BivariateCopula`, which defines the common interface for CDF,
PDF, log-PDF, and log-likelihood evaluation.
"""

from .base import BivariateCopula
from .frank import FrankCopula

__all__ = [
    "BivariateCopula",
    "FrankCopula",
]
