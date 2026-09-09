"""Estimate the dependence parameter of a Frank copula by maximum likelihood.

This example illustrates a complete bivariate fitting workflow:

1. define two paired normalized samples ``u`` and ``v``;
2. define the negative log-likelihood objective;
3. optimize the objective over the positive and negative Frank parameter
   domains separately; and
4. select the parameter estimate with the smallest negative log-likelihood.

The Frank independence case corresponds to the limiting value ``theta -> 0``.
Because ``theta = 0`` is excluded from the numerical search, the optimization
is performed on intervals immediately to either side of zero.
"""

import numpy as np

from pycopula_ml.copulas import FrankCopula
from pycopula_ml.estimation import fit_copula_mle

def negative_log_likelihood(theta, u, v):
    """Return the negative Frank copula log-likelihood for a candidate theta.

    Parameters
    ----------
    theta : float
        Candidate Frank dependence parameter. Values close to zero represent
        the limiting independence case, but zero itself is not evaluated.
    u : numpy.ndarray
        Pseudo-observations for the first variable, with values in ``(0, 1)``.
    v : numpy.ndarray
        Pseudo-observations for the second variable, with values in ``(0, 1)``.

    Returns
    -------
    float
        Negative log-likelihood. Minimizing this quantity is equivalent to
        maximizing the copula log-likelihood.
    """
    copula = FrankCopula(theta=theta)
    return -copula.log_likelihood(u, v)


if __name__ == "__main__":
    """Estimate theta and report the direction of the fitted association."""
    u = np.array([
        0.423077, 0.807692, 0.346154, 0.500000, 0.384615,
        0.730769, 0.461538, 0.076923, 0.192308, 0.923077,
        0.884615, 0.230769, 0.576923, 0.269231, 0.692308,
        0.769231, 0.038462, 0.846154, 0.538462, 0.307692,
        0.153846, 0.115385, 0.653846, 0.615385, 0.961538
    ])

    v = np.array([
        0.538462, 0.769231, 0.461538, 0.346154, 0.615385,
        0.500000, 0.653846, 0.076923, 0.153846, 0.846154,
        0.384615, 0.423077, 0.576923, 0.115385, 0.923077,
        0.807692, 0.307692, 0.884615, 0.961538, 0.192308,
        0.230769, 0.038462, 0.269231, 0.730769, 0.692308
    ])

    result = fit_copula_mle(
        FrankCopula,
        u,
        v,
        bounds=(
            (-50.0, -1e-6),
            (1e-6, 50.0),
        ),
    )

    theta_hat = result.theta

    if np.isclose(theta_hat, 0.0, atol=1e-4):
        association = "approximately independent"
    elif theta_hat > 0:
        association = "positive"
    else:
        association = "negative"

    print(f"Association: {association}")
    print(f"theta_hat: {theta_hat}")
    print(f"log-likelihood: {-result.log_likelihood}")