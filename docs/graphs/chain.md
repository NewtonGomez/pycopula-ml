# Maximum-mutual-information chain

## Objective

Given a symmetric, nonnegative pairwise mutual-information matrix
\(M\in\mathbb{R}^{d\times d}\), select an open path visiting all features
exactly once and maximizing

\[
\max_{\pi\in S_d}
\sum_{k=1}^{d-1}M_{\pi_k,\pi_{k+1}}.
\]

The chain has \(d-1\) undirected edges and does not branch. All weights
are measured in **nats**. Finding the global optimum is computationally
hard; the supported algorithms are **heuristics**, not exact solvers.

## API

```python
from pycopula_ml.graphs import build_chain_graph

chain = build_chain_graph(mi_matrix, method="best")
```

`mi_matrix` may be a NumPy array or a `CopulaWeightMatrices` result. In the
latter case, the builder extracts `.mutual_information` rather than the
configurable `.weights` property.

| Member | Meaning |
| --- | --- |
| `order` | Original zero-based feature indices in chain order |
| `permutation` | Feature names in chain order |
| `edges` | Named edges and their mutual-information weights |
| `total_mi` | Sum of the selected weights (nats) |
| `method` | Selected algorithm (`"greedy"` or `"tsp"`) |
| `adjacency_matrix(weighted=True)` | Symmetric weighted adjacency |
| `edge_index(bidirectional=True)` | NumPy indices for graph algorithms |
| `edge_attributes(bidirectional=True)` | Edge weights aligned with indices |

The input must be square, finite, symmetric, have a zero diagonal, and have
nonnegative entries. Feature names must be unique when supplied.

## Methods

- **`"greedy"`:** Start from the strongest pair and extend either endpoint
  with the strongest available connection.
- **`"tsp"`:** Use multiple nearest-neighbor starts and 2-opt improvements
  to construct a high-scoring open path.
- **`"best"`:** Run both heuristics and return the higher-scoring path.

A chain differs from a [maximum-weight spanning tree](mst.md), which may
have branching vertices. If an algorithm needs the unweighted topology,
use `adjacency_matrix(weighted=False)` to preserve zero-weight edges.

## Use with classification data

Estimate marginal transformations and dependence on training records only.
For class-specific models, compute separate matrices within each training
class. The resulting chain represents statistical feature dependence; it
neither proves causality nor performs class prediction.

See the [chain example](../examples/chain.md).
