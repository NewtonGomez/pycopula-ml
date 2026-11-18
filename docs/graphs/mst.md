# Maximum-mutual-information spanning tree (MST)

## Problem and terminology

Given a symmetric pairwise mutual-information matrix
$M\in\mathbb{R}^{d\times d}$, select a set of $d-1$ undirected edges that
connects all $d$ variables, has no cycles, and **maximizes** the sum of
selected mutual-information weights:

$$
T^* = \underset{T\text{ spanning tree}}{\arg\max}
\sum_{(i,j)\in E(T)} M_{ij}.
$$

Here $M_{ij}=I(X_i;X_j)\geq 0$ is measured in **nats**. This is a
**maximum-weight spanning tree**, obtained using a *minimum spanning tree*
algorithm on transformed costs:

$$
C_{ij} = -M_{ij}, \qquad
\underset{T}{\arg\min}\sum_{(i,j)\in E(T)}C_{ij}
= \underset{T}{\arg\max}\sum_{(i,j)\in E(T)}M_{ij}.
$$

**Important:** callers provide the **positive MI matrix**. The Python
implementation multiplies each weight by **-1 internally**. Do not negate the
input yourself. The result reports both `total_mi` (maximized) and
`total_cost` (minimized, equal to `-total_mi`).

A spanning tree is *not* a Hamiltonian chain. Trees can branch; a chain
requires all vertices to have degree at most two. The maximum-MI tree is
therefore never worse in total selected MI than the optimum MI chain on the
same matrix, because every chain is also a tree.

## Exact algorithm: Kruskal on negative MI

1. Validate that the matrix is symmetric, finite, square, has zero diagonal,
   and contains nonnegative MI values.
2. Create all unordered candidate pairs $(i,j)$ with $i<j$.
3. Set the cost of every pair to $-M_{ij}$ and sort **ascending** by cost,
   breaking ties by feature indices $(i,j)$.
4. Traverse the sorted pairs. Keep an edge only if it connects two
   previously separate components (union–find / disjoint-set structure).
5. Stop after selecting exactly $d-1$ edges.

Kruskal solves the spanning-tree objective **globally** for the supplied
weights, unlike the greedy and TSP-inspired algorithms used for the more
constrained chain problem. If multiple trees have the same optimum score,
our index-based tie-breaking makes results deterministic; it is not a
statement about the uniqueness of the statistical dependence structure.

For a dense matrix, creating and sorting all pairs takes $O(d^2\log d)$
time and $O(d^2)$ additional memory. This excludes the potentially more
expensive work of estimating copulas and calculating their MI values.

## API and example

```python
import numpy as np

from pycopula_ml.graphs import build_mst_graph

mi = np.array(
    [
        [0, 1, 8, 2],
        [1, 0, 3, 9],
        [8, 3, 0, 7],
        [2, 9, 7, 0],
    ],
    dtype=float,
)

mst = build_mst_graph(mi, feature_names=["A", "B", "C", "D"])

print(mst.edge_indices)  # ((1, 3), (0, 2), (2, 3))
print(mst.total_mi)  # 24.0
print(mst.total_cost)  # -24.0
print(mst.edge_weights)  # [9. 8. 7.]
print(mst.edge_costs)  # [-9. -8. -7.]

adjacency = mst.adjacency_matrix()
edge_index = mst.edge_index()
edge_attr = mst.edge_attributes()
```

This matrix happens to give the same *topology* as the example chain:
`A--C--D--B`, but that coincidence is not guaranteed.

For example, if one variable has MI 9, 8, and 7 with three other variables,
and all three leaf-to-leaf MI values are 1, the optimal tree is a star
with score 24, whereas a chain cannot use all three central edges.

## Input and output contracts

`build_mst_graph(mi_matrix, *, feature_names=None)` accepts either a
`numpy`-compatible matrix or a `CopulaWeightMatrices` object. In the latter
case it **always** reads `.mutual_information`, not `.weights` (which can
contain pairwise log-likelihoods instead).

`SpanningTreeGraph` contains:

| Member | Interpretation |
| --- | --- |
| `feature_names` | Original column order and labels |
| `edge_indices` | Selected undirected pairs `(i,j)` with `i < j` |
| `edges` | `SpanningTreeEdge(source, target, mutual_information)` |
| `edge_weights` | Positive MI on selected edges, in nats |
| `edge_costs` | Negative MI used to choose the tree |
| `total_mi` | Sum of the MI values; maximized |
| `total_cost` | Sum of negative MI costs; minimized |
| `adjacency_matrix(weighted=True)` | Symmetric MI-weighted adjacency |
| `adjacency_matrix(weighted=False)` | Binary topology, including zero-MI edges |
| `edge_index(bidirectional=True)` | `(2, 2(d-1))` edge-index array |
| `edge_attributes(bidirectional=True)` | `(2(d-1), 1)` MI features aligned with edge-index columns |

For interoperability with graph-processing code, undirected edges
appear in both directions in `edge_index()` by default. They still represent
one mathematical edge per pair. `edge_attributes()` follows the same order.
No separate graph-learning framework is required.

### Zero-valued MI

Zero MI means **zero statistical weight**, not "edge forbidden". The input
is a complete candidate graph. If all MI values are zero, the algorithm
returns a valid spanning tree with $d-1$ edges and a total MI of zero.
Use `adjacency_matrix(weighted=False)` to inspect the selected topology:
zero-weight edges cannot be distinguished from absent edges in the weighted
adjacency matrix.

### Connection to the copula pipeline

```python
from pycopula_ml.graphs import build_mst_graph

# `matrices` is a CopulaWeightMatrices result computed earlier.
mst = build_mst_graph(matrices)
```

The copulas and marginal transformations are fitted **before** MST selection.
In a supervised classifier, fit these transformations on training data only,
and avoid leakage from validation/test data.

A maximum-MI spanning tree provides a feature-dependence topology.
Its edges do **not** imply causality or prove conditional independence in
the original data. Model-based MI inherits the uncertainty of the fitted
copulas and marginal transformations.

See also [MST usage example](../examples/mst.md) and
[graph validation](../testing/graph_validation.md).
