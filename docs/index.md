# pycopula-ml documentation

This directory documents the current architecture, mathematical foundations,
implemented API, testing strategy, and development roadmap of
**pycopula-ml**.

The current statistical workflow has three main stages:

$$
X,Y
\rightarrow
U,V
$$

through marginal modeling or rank-based pseudo-observations,

$$
(U,V,\theta)
\rightarrow
C_\theta(u,v)
$$

through a bivariate copula, and

$$
(U,V)
\rightarrow
\widehat{\theta}
$$

through numerical parameter estimation.

The current implementation includes:

- marginal models and marginal model selection;
- rank-based pseudo-observations;
- the bivariate Frank copula;
- copula CDF, density, log-density, and log-likelihood evaluation;
- scalar maximum-likelihood estimation through `fit_copula_mle()`;
- support for disjoint parameter-search intervals;
- deterministic unit tests and statistical validation workflows.

## Contents

### Getting Started

- [Development workflow](getting-started/development.md)

### Concepts

- [Mathematical foundations](concepts/mathematical_foundations.md)
- [Copulas](concepts/copulas.md)
- [Marginal models](concepts/marginals.md)
- [Parameter estimation](concepts/estimation.md)

### Architecture

- [Project architecture](architecture/architecture.md)

### Copulas

- [Frank copula](copulas/frank_copula.md)

### Marginals

- [Empirical marginals](marginals/empirical_marginals.md)
- [Parametric marginals](marginals/parametric_marginals.md)
- [Marginal model selection](marginals/model_selection.md)

### Estimation

- [Maximum-likelihood estimation](estimation/maximum_likelihood.md)

### Testing and Validation

- [Estimation testing and statistical validation](testing/estimation_validation.md)

### API Reference

- [Copula API](api/copulas.md)
- [Marginal API](api/marginals.md)
- [Estimation API](api/estimation.md)

### References

- [References](references/references.md)
