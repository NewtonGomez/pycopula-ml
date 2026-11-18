# pycopula-ml

**pycopula-ml** is a Python library for bivariate copula modeling,
statistical dependence analysis, and graph construction. Its components can be
used to develop probabilistic classification pipelines with continuous
features.

The current package provides copula families, marginal distributions,
parameter estimation, model selection, mutual information, and graph-building
utilities. **It does not expose a complete `fit()` / `predict()` classifier or
train a neural network.**

## Features

- **Six bivariate copula families:** Frank, Clayton, Gaussian, Joe,
  Gumbel-Hougaard, and Ali-Mikhail-Haq.
- **Marginal transformations:** continuous parametric marginals, empirical
  marginals, rank-based pseudo-observations, and candidate selection.
- **Parameter estimation:** bounded maximum likelihood, including disjoint
  search intervals.
- **Model selection:** AIC and BIC for fitted copula candidates.
- **Dependence analysis:** copula-based mutual information and pairwise
  dependence matrices.
- **Graph construction:** maximum-mutual-information spanning trees (MST) and
  heuristic maximum-mutual-information chains.

## Statistical workflow

For a feature pair \((X,Y)\), marginal transforms produce uniform
pseudo-observations \((U,V)\). A bivariate copula is then fitted by maximizing

\[
\widehat\theta = \operatorname*{argmax}_{\theta}
\sum_{i=1}^{n} \log c_\theta(u_i,v_i).
\]

A fitted copula also provides model-based mutual information:

\[
I_C(X;Y) = \int_0^1\!\int_0^1 c(u,v)\log c(u,v)\,du\,dv.
\]

For multiple features, pairwise estimates yield a dependence matrix that can
be used to construct an MST or a chain. When fitting models per class, **only
training observations from that class** should be used.

## Documentation

- [Quick start](getting-started/quickstart.md)
- [Development workflow](getting-started/development.md)
- [Mathematical foundations](concepts/mathematical_foundations.md)
- [Copulas](concepts/copulas.md)
- [Marginals](concepts/marginals.md)
- [Parameter estimation](concepts/estimation.md)
- [Dependence measures](concepts/dependence.md)
- [Architecture](architecture/architecture.md)
- [Copula families](copulas/parametrics/index.md)
- [Marginal model selection](marginals/model_selection.md)
- [Copula model selection](model-selection/selection.md)
- [Pairwise weight matrices](dependence/weight_matrices.md)
- [Chain construction](graphs/chain.md)
- [Maximum spanning tree](graphs/mst.md)
- [Examples](examples/dependence.md)
- [API reference](api/copulas.md)
- [Testing and validation](testing/estimation_validation.md)
- [References](references/references.md)

The installed package version remains the authority for available exports and
accepted parameters.
