# Example: exact maximum-MI spanning tree

This example illustrates how a minimum spanning tree algorithm applied to
**negative** mutual information maximizes the retained information.

```python
import numpy as np

from pycopula_ml.graphs import build_chain_graph, build_mst_graph

mi = np.array(
    [
        [0, 9, 8, 7],
        [9, 0, 1, 1],
        [8, 1, 0, 1],
        [7, 1, 1, 0],
    ],
    dtype=float,
)

labels = ["central", "left", "middle", "right"]

mst = build_mst_graph(mi, feature_names=labels)
chain = build_chain_graph(mi, method="best", feature_names=labels)

print(mst.edge_indices)  # ((0, 1), (0, 2), (0, 3))
print(mst.total_mi)  # 24.0
print(mst.total_cost)  # -24.0
print(chain.total_mi)  # Less than 24: a chain cannot branch
print(mst.adjacency_matrix(weighted=False))

# Matrices for downstream graph-learning code:
edge_index = mst.edge_index()  # (2, 6), both directions
edge_attributes = mst.edge_attributes()  # (6, 1), positive MI values
```

## Connecting a `CopulaWeightMatrices` result

After computing pseudo-observations and fitting bivariate copulas with
`build_weight_matrices`, you may pass the result object directly:

```python
from pycopula_ml.graphs import build_mst_graph

# `matrices` is a CopulaWeightMatrices result computed earlier.
mst = build_mst_graph(matrices)
```

`build_mst_graph` extracts the `.mutual_information` matrix **even if**
`matrices.weight_method == "likelihood"`. You should not negate the MI
matrix before passing it; `-MI` is an internal **optimization cost**.

Note that an optimal tree can have maximum degree greater than two; use
[chain construction](chain.md) if your model explicitly requires a path.
