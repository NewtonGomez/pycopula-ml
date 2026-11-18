# Example: fit Gaussian copulas and construct an MI chain

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices
from pycopula_ml.graphs import build_chain_graph

rng = np.random.default_rng(2026)
z = rng.normal(size=200)
x = np.column_stack(
    [
        z + rng.normal(scale=0.2, size=200),
        rng.normal(size=200),
        z + rng.normal(scale=0.3, size=200),
        rng.normal(size=200),
    ]
)

# Training data only: convert each marginal to pseudo-observations in (0,1).
u = rankdata(x, axis=0, method="average") / (len(x) + 1)

weights = build_weight_matrices(
    u,
    GaussianCopula,
    bounds=(-0.98, 0.98),
    weight_method="mutual_information",
    feature_names=("feature_a", "feature_b", "feature_c", "feature_d"),
)
chain = build_chain_graph(weights, method="best")

for edge in chain.edges:
    print(edge.source, "--", edge.target, edge.mutual_information)
print("Chain:", chain.permutation)
print("Total MI (nats):", chain.total_mi)

# Compatible edge indices and attributes for a graph-learning pipeline:
edge_index = chain.edge_index()
edge_attr = chain.edge_attributes()
assert edge_attr.shape[0] == edge_index.shape[1]
```

The example uses synthetic observations and generic feature names.
Fitted copulas describe statistical dependence; the graph constructor
selects a path from the mutual-information matrix.
