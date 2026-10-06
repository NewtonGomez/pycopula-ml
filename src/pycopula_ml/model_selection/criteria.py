"""Information criteria for statistical model comparison."""

from __future__ import annotations

import math
from numbers import Integral, Real


def _validate_log_likelihood(log_likelihood: Real) -> float:
    """Validate and normalize a log-likelihood value."""
    value = float(log_likelihood)

    if not math.isfinite(value):
        raise ValueError("log_likelihood must be a finite number.")

    return value


def _validate_n_parameters(n_parameters: Integral) -> int:
    """Validate the number of free parameters in a model."""
    if isinstance(n_parameters, bool) or not isinstance(n_parameters, Integral):
        raise TypeError("n_parameters must be an integer.")

    value = int(n_parameters)

    if value < 0:
        raise ValueError("n_parameters must be greater than or equal to 0.")

    return value


def _validate_n_observations(n_observations: Integral) -> int:
    """Validate the number of observations used to fit a model."""
    if isinstance(n_observations, bool) or not isinstance(
        n_observations,
        Integral,
    ):
        raise TypeError("n_observations must be an integer.")

    value = int(n_observations)

    if value <= 0:
        raise ValueError("n_observations must be greater than 0.")

    return value


def aic(log_likelihood: Real, n_parameters: Integral) -> float:
    """Compute the Akaike Information Criterion.

    The Akaike Information Criterion is defined as

    .. math::

        \mathrm{AIC} = 2k - 2\ell(\hat{\theta}),

    where ``k`` is the number of free model parameters and
    :math:`\ell(\hat{\theta})` is the maximized log-likelihood.

    Lower AIC values indicate a preferred model among models fitted to the
    same observations.

    Parameters
    ----------
    log_likelihood:
        Maximized log-likelihood of the fitted model.
    n_parameters:
        Number of free parameters estimated in the model.

    Returns
    -------
    float
        Akaike Information Criterion.
    """
    log_likelihood = _validate_log_likelihood(log_likelihood)
    n_parameters = _validate_n_parameters(n_parameters)

    return 2.0 * n_parameters - 2.0 * log_likelihood


def bic(
    log_likelihood: Real,
    n_parameters: Integral,
    n_observations: Integral,
) -> float:
    """Compute the Bayesian Information Criterion.

    The Bayesian Information Criterion is defined as

    .. math::

        \mathrm{BIC}
        = k\log(n) - 2\ell(\hat{\theta}),

    where ``k`` is the number of free model parameters, ``n`` is the number
    of observations, and :math:`\ell(\hat{\theta})` is the maximized
    log-likelihood.

    For bivariate copula data, one observation corresponds to one pair
    :math:`(u_i, v_i)`. Therefore, when ``u`` and ``v`` have the same length,
    ``n_observations`` is ``len(u)`` rather than ``len(u) + len(v)``.

    Lower BIC values indicate a preferred model among models fitted to the
    same observations.

    Parameters
    ----------
    log_likelihood:
        Maximized log-likelihood of the fitted model.
    n_parameters:
        Number of free parameters estimated in the model.
    n_observations:
        Number of observations used to fit the model.

    Returns
    -------
    float
        Bayesian Information Criterion.
    """
    log_likelihood = _validate_log_likelihood(log_likelihood)
    n_parameters = _validate_n_parameters(n_parameters)
    n_observations = _validate_n_observations(n_observations)

    return (
        n_parameters * math.log(n_observations)
        - 2.0 * log_likelihood
    )
