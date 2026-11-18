# pycopula-ml

**Copula-based dependence modeling and probabilistic machine-learning utilities for Python.**

`pycopula-ml` provides statistical building blocks for modeling dependence
between continuous features, selecting bivariate copula families, and
constructing feature-dependence graphs. These tools can be integrated into
probabilistic classification workflows without tying the library to any
particular application domain.

> **Project status:** Development / alpha. The documented modules provide
> copula modeling, estimation, dependence analysis, and graph construction.
> A complete classification estimator with a `fit()` / `predict()` interface
> is **not** part of the documented public API.

## Features

- **Parametric copulas:** Frank, Clayton, Gaussian, Joe, Gumbel-Hougaard,
  and Ali-Mikhail-Haq.
- **Common copula API:** cumulative distribution (`cdf`), density (`pdf`),
  log-density (`logpdf`), and sample log-likelihood (`log_likelihood`).
- **Marginal modeling:** parametric and empirical marginals, marginal
  selection, and rank-based pseudo-observations.
- **Parameter estimation:** bounded maximum-likelihood estimation, including
  disjoint parameter intervals.
- **Model selection:** comparison of copula candidates using AIC and BIC.
- **Dependence analysis:** model-based mutual information and pairwise
  parameter, log-likelihood, and mutual-information matrices.
- **Graph construction:** maximum-mutual-information spanning trees and
  heuristic maximum-mutual-information chains.

The core statistical functionality is documented around **NumPy** and
**SciPy**. Graph utilities do not require a deep-learning framework.

## Installation

Clone the repository and install from the project root:

```bash
git clone https://github.com/NewtonGomez/pycopula-ml.git
cd pycopula-ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment using
`.venv\Scripts\activate` instead of `source`.

For a development installation, use the dependency group provided by the
repository:

```bash
python -m pip install -e ".[dev]"
```

## Quick example: fit a copula

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas import GaussianCopula
from pycopula_ml.dependence import mutual_information
from pycopula_ml.estimation import fit_copula_mle

rng = np.random.default_rng(42)
x = rng.normal(size=200)
y = 0.7 * x + rng.normal(scale=0.7, size=200)

# Transform paired observations to the open unit interval.
u = rankdata(x, method="average") / (len(x) + 1)
v = rankdata(y, method="average") / (len(y) + 1)

fit = fit_copula_mle(
    GaussianCopula,
    u,
    v,
    bounds=(-0.98, 0.98),
)

copula = GaussianCopula(theta=fit.theta)
information = mutual_information(copula)

print(f"Copula parameter: {fit.theta:.4f}")
print(f"Mutual information: {information:.4f} nats")
```

The pseudo-observations must represent matched rows from the original
features. Estimation bounds must be chosen for the selected copula family.

## Feature-dependence matrices and graphs

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices
from pycopula_ml.graphs import build_mst_graph

rng = np.random.default_rng(123)
features = rng.normal(size=(200, 4))
u = rankdata(features, axis=0, method="average") / (len(features) + 1)

matrices = build_weight_matrices(
    u,
    GaussianCopula,
    bounds=(-0.98, 0.98),
    weight_method="mutual_information",
    feature_names=("feature_a", "feature_b", "feature_c", "feature_d"),
)

tree = build_mst_graph(matrices)
print(matrices.mutual_information)
print(tree.edge_indices)
print(tree.total_mi)
```

`build_mst_graph()` maximizes the total weight of a spanning tree for the
provided mutual-information matrix. For a nonbranching path, use
`build_chain_graph()` instead. Neither structure is a trained classifier.

## Using the components in classification

A class-conditional dependence workflow typically consists of:

1. Splitting labeled observations into training and evaluation partitions.
2. Estimating marginal transformations using **training data only**.
3. Fitting a copula for each relevant feature pair, optionally separately
   within each training class.
4. Constructing dependence matrices or feature graphs.
5. Implementing and validating a class-scoring and prediction rule in the
   consuming application.

The package documents steps 2–4 and the underlying statistical utilities;
step 5 is not offered as a complete estimator by the documented public API.
Always avoid data leakage between training and evaluation partitions.

## Documentation

Full documentation: https://newtongomez.github.io/pycopula-ml/

- [Quick start](docs/getting-started/quickstart.md)
- [Mathematical foundations](docs/concepts/mathematical_foundations.md)
- [Parametric copulas](docs/copulas/parametrics/index.md)
- [Marginal models](docs/concepts/marginals.md)
- [Maximum-likelihood estimation](docs/estimation/maximum_likelihood.md)
- [Model selection](docs/model-selection/selection.md)
- [Mutual information](docs/dependence/mutual_information.md)
- [Pairwise weight matrices](docs/dependence/weight_matrices.md)
- [Maximum-MI spanning tree](docs/graphs/mst.md)
- [Maximum-MI chain](docs/graphs/chain.md)
- [API reference](docs/api/copulas.md)

### Local documentation preview

```bash
python -m pip install mkdocs-material
python -m mkdocs serve
```

To check the site build:

```bash
python -m mkdocs build --strict
```

## Development and tests

Use PEP 8 conventions for Python code, public interfaces, and examples. The
project documents Ruff for formatting and linting:

```bash
ruff format .
ruff check .
python -m pytest
```

See [development workflow](docs/getting-started/development.md) and
[testing guidance](docs/testing/estimation_validation.md) for details.

## Scope and limitations

- Implemented copulas are documented as **bivariate** families. Do not infer
  multivariate copula support from the feature-graph utilities.
- Mutual-information values are model-based estimates. They are not causal
  relationships or proof of conditional independence.
- Chain construction uses heuristic search; maximum spanning trees optimize
  the spanning-tree objective for the supplied weights.
- Input requirements and public exports may change while the project remains
  in development.

## Contributing

Contributions should include focused tests, public API documentation, and
reproducible examples. Before opening a pull request, run the formatter,
linter, and relevant test suite. Keep library examples independent of any
specific scientific or industry dataset.

## Project links

- **Repository:** https://github.com/NewtonGomez/pycopula-ml
- **Documentation:** https://newtongomez.github.io/pycopula-ml/

See the repository's actual license terms, if provided, before reuse.
