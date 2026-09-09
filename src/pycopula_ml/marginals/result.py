"""
Result containers for marginal distribution fitting.

This module defines structured result objects used to store the statistical
diagnostics produced when fitting parametric marginal distributions.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MarginalFitResult:
    """
    Store the result of fitting a marginal probability distribution.

    Parameters
    ----------
    distribution : str
        Name assigned to the fitted probability distribution.
    parameters : tuple of float
        Estimated SciPy distribution parameters in the order returned by
        ``distribution.fit``.
    sample_size : int
        Number of observations used for fitting.
    n_parameters : int
        Number of free parameters estimated from the sample.
    log_likelihood : float
        Maximized sample log-likelihood.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    ks_statistic : float
        Kolmogorov-Smirnov distance between the empirical and fitted CDF.
    ks_pvalue : float
        Exploratory Kolmogorov-Smirnov p-value.

    Notes
    -----
    When distribution parameters are estimated from the same sample used in
    the Kolmogorov-Smirnov test, the classical KS p-value is not formally
    calibrated. It is stored as an exploratory diagnostic and should not be
    used as the default model-selection criterion.
    """

    distribution: str
    parameters: Tuple[float, ...]
    sample_size: int
    n_parameters: int
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float
