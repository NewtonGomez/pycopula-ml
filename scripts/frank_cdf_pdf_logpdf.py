"""Evaluate the CDF, PDF, and log-PDF of a Frank copula.

This example creates a bivariate :class:`FrankCopula` with a fixed dependence
parameter ``theta`` and evaluates three fundamental quantities at a point
``(u, v)`` in the unit square:

- the cumulative distribution function (CDF), ``C(u, v)``;
- the copula density (PDF), ``c(u, v)``;
- the natural logarithm of the density, ``log(c(u, v))``.

The script is intended as a minimal usage example for inspecting the behavior
of a fitted or manually specified Frank copula.
"""

from pycopula_ml.copulas.frank import FrankCopula


def main():
    """Create a Frank copula and evaluate its main distribution functions."""
    copula = FrankCopula(theta=2.5)

    u = 0.3
    v = 0.7

    cdf_value = copula.cdf(u, v)
    pdf_value = copula.pdf(u, v)
    logpdf_value = copula.logpdf(u, v)

    print(f"CDF: {cdf_value}")
    print(f"PDF: {pdf_value}")
    print(f"log-PDF: {logpdf_value}")


if __name__ == "__main__":
    main()
