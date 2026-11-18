# Graph construction API

## `build_chain_graph()`

```python
from pycopula_ml.graphs import build_chain_graph

build_chain_graph(
    mi_matrix,
    *,
    method="best",
    feature_names=None,
) -> ChainGraph
```

- `mi_matrix`: `(d,d)` symmetric MI matrix or `CopulaWeightMatrices`.
- `method`: `"greedy"`, `"tsp"`, or `"best"`.
- `feature_names`: unique nonempty labels. If `None`, infer names from
  `CopulaWeightMatrices`, otherwise use `X1`, `X2`, ... .
- Returns an immutable-result `ChainGraph` object. Its returned NumPy arrays
  are new allocations.
- Raises `ValueError` for invalid method, dimensions, symmetry, negativity,
  diagonal, non-finite values, or names.

The score is in nats. No method guarantees a globally optimal Hamiltonian
path. For mathematical details, read [Chain graphs](../graphs/chain.md).

## `ChainGraph`

`order`, `permutation`, `edges`, `total_mi`, `method`, `feature_names`,
`edge_weights`, `adjacency_matrix(weighted=True)`, `edge_index()`, and
`edge_attributes()`.

## `ChainEdge`

Frozen dataclass with `source: str`, `target: str`, and
`mutual_information: float` (nats).
