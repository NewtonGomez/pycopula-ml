"""Estimate the dependence parameter of a Frank copula by maximum likelihood.

This example illustrates a complete bivariate fitting workflow:

1. define two paired samples ``x`` and ``y``;
2. transform them into pseudo-observations ``u`` and ``v``;
3. maximize the Frank copula log-likelihood over the negative and positive
   parameter domains; and
4. report the estimated dependence parameter and association direction.

The Frank independence case corresponds to the limiting value ``theta -> 0``.
Because ``theta = 0`` is excluded from the numerical search, both sides of
zero are searched separately.
"""

import numpy as np

from pycopula_ml.copulas import FrankCopula
from pycopula_ml.estimation import fit_copula_mle
from pycopula_ml.marginals import bivariate_pseudo_observations


def main() -> None:
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
    elif theta_hat > 0.0:
        association = "positive"
    else:
        association = "negative"

    print(f"Association: {association}")
    print(f"theta_hat: {theta_hat:.8f}")
    print(f"log-likelihood: {result.log_likelihood:.8f}")
    print(f"observations: {result.n_obs}")
    print(f"selected interval: {result.interval}")
    print(f"optimizer evaluations: {result.nfev}")


if __name__ == "__main__":
    main()
