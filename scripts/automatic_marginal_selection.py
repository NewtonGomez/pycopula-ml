"""Select and apply a univariate marginal distribution automatically.

This example demonstrates the :class:`MarginalSelector` workflow for a single
sample. The selector fits the available candidate marginal distributions,
ranks them using its configured model-selection criterion, reports the fitting
summary, and transforms the original observations to the unit interval using
the CDF of the selected marginal model.

The transformed values can subsequently be used as copula inputs when a
parametric marginal transformation is preferred over rank-based
pseudo-observations.
"""

import numpy as np

from pycopula_ml.marginals import MarginalSelector


def main():
    """Fit the marginal selector and transform the sample to the unit interval."""
    x = np.array(
        [
            2.9499,
            7.8283,
            5.5270,
            2.3538,
            0.4862,
            3.4950,
        ]
    )

    selector = MarginalSelector()
    selector.fit(x)

    print(selector.summary())

    u = selector.transform(x)

    print()
    print("Transformed observations:")
    print(u)


if __name__ == "__main__":
    main()
