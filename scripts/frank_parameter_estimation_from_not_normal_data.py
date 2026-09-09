"""Estimate the dependence parameter of a Frank copula by maximum likelihood.

This example illustrates a complete bivariate fitting workflow:

1. define two paired samples ``x`` and ``y``;
2. transform them into pseudo-observations ``u`` and ``v``;
3. define the negative log-likelihood objective;
4. optimize the objective over the positive and negative Frank parameter
   domains separately; and
5. select the parameter estimate with the smallest negative log-likelihood.

The Frank independence case corresponds to the limiting value ``theta -> 0``.
Because ``theta = 0`` is excluded from the numerical search, the optimization
is performed on intervals immediately to either side of zero.
"""

import numpy as np
from scipy import optimize

from pycopula_ml.copulas import FrankCopula
from pycopula_ml.marginals import bivariate_pseudo_observations


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


def main():
    """Estimate theta and report the direction of the fitted association."""
    x = np.array(
        [
            17.152,
            22.891,
            13.119,
            18.684,
            14.785,
            22.713,
            17.709,
            6.323,
            10.450,
            24.125,
            24.041,
            10.847,
            19.822,
            11.365,
            22.517,
            22.726,
            5.720,
            23.232,
            19.557,
            11.404,
            9.823,
            7.014,
            21.608,
            19.955,
            24.762,
        ]
    )

    y = np.array(
        [
            28.706,
            35.109,
            25.353,
            23.579,
            30.108,
            26.524,
            30.365,
            9.774,
            14.288,
            37.614,
            24.586,
            24.835,
            29.213,
            13.084,
            40.858,
            36.340,
            20.746,
            38.858,
            53.280,
            16.110,
            19.457,
            3.990,
            20.256,
            34.827,
            30.668,
        ]
    )

    u, v = bivariate_pseudo_observations(x, y)

    result_positive = optimize.minimize_scalar(
        negative_log_likelihood,
        args=(u, v),
        bounds=(1e-6, 50.0),
        method="bounded",
    )
    result_negative = optimize.minimize_scalar(
        negative_log_likelihood,
        args=(u, v),
        bounds=(-50.0, -1e-6),
        method="bounded",
    )

    best_result = min(
        (result_positive, result_negative),
        key=lambda result: result.fun,
    )
    theta_hat = best_result.x

    if np.isclose(theta_hat, 0.0, atol=1e-4):
        association = "approximately independent"
    elif theta_hat > 0:
        association = "positive"
    else:
        association = "negative"

    print(f"Association: {association}")
    print(f"theta_hat: {theta_hat}")
    print(f"log-likelihood: {-best_result.fun}")


if __name__ == "__main__":
    main()
