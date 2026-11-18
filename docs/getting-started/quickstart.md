# Quick start

## Installation

From the repository root, create a virtual environment and install the package
in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Fit a bivariate copula

The copula fitting functions expect paired observations in the open unit
interval. Convert a pair of continuous feature columns to rank-based
pseudo-observations before estimating dependence.

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas import GaussianCopula
from pycopula_ml.dependence import mutual_information
from pycopula_ml.estimation import fit_copula_mle

rng = np.random.default_rng(42)
x = rng.normal(size=200)
y = 0.7 * x + rng.normal(scale=0.7, size=200)

u = rankdata(x, method="average") / (len(x) + 1)
v = rankdata(y, method="average") / (len(y) + 1)

fit = fit_copula_mle(
    GaussianCopula,
    u,
    v,
    bounds=(-0.98, 0.98),
)

copula = GaussianCopula(theta=fit.theta)
print(f"Estimated parameter: {fit.theta:.4f}")
print(f"Mutual information: {mutual_information(copula):.4f} nats")
```

## Build a feature-dependence graph

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices
from pycopula_ml.graphs import build_mst_graph

rng = np.random.default_rng(42)
features = rng.normal(size=(200, 4))
u = rankdata(features, axis=0, method="average") / (len(features) + 1)

matrices = build_weight_matrices(
    u,
    GaussianCopula,
    bounds=(-0.98, 0.98),
    feature_names=("feature_a", "feature_b", "feature_c", "feature_d"),
)
graph = build_mst_graph(matrices)

print(graph.total_mi)
print(graph.edge_indices)
```

For class-specific dependence models, fit marginal transforms, copulas, and
feature graphs exclusively on each class's **training data**. The graph is a
structural component; it is not itself a classifier.

Continue with the [dependence examples](../examples/dependence.md) or the
[API reference](../api/copulas.md).
