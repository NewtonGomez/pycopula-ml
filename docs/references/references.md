# References

## Copulas and bivariate distributions

Balakrishnan, N., & Lai, C.-D. (2009).  
*Continuous Bivariate Distributions, Second Edition*. Springer.

Relevant topics include:

- bivariate copulas;
- continuous bivariate distributions;
- Frank distribution/copula;
- dependence modeling.

## Copula graphical neural networks

The project also uses the CopulaGNN literature as broader context for combining
copulas, dependence modeling, and graph-based machine learning.

## MLCOPULA

The semester project is motivated by the R package **MLCOPULA** and the
objective of reproducing and modernizing its copula-based classification
workflow in Python.

As implementation parity progresses, this documentation should record:

- R source function;
- corresponding Python component;
- whether behavior is directly reproduced or intentionally redesigned;
- numerical differences;
- test fixtures;
- version of the R package used as reference.

## SciPy

SciPy provides the numerical and statistical backend used by several parts of
the package.

Marginal modeling relies on functionality such as:

- `distribution.fit`;
- `distribution.pdf`;
- `distribution.logpdf`;
- `distribution.cdf`;
- `distribution.support`;
- `scipy.stats.kstest`;
- `scipy.stats.rankdata`.

The parameter-estimation layer currently relies internally on:

- `scipy.optimize.minimize_scalar`.

The public package API intentionally wraps these capabilities instead of
exposing SciPy implementation details as part of the main user workflow.

## Estimation terminology

For copula fitting, the package documentation distinguishes between:

- likelihood based on exact transformed uniforms under known marginals; and
- pseudo-likelihood based on estimated or rank-derived pseudo-observations.

The numerical objective has the same computational form,

$$
\ell(\theta)
=
\sum_i \log c_\theta(u_i,v_i),
$$

but the statistical interpretation depends on how \(u_i,v_i\) were obtained.
