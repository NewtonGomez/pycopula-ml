"""Maximum-mutual-information spanning trees via minimum spanning trees.

Kruskal's *minimum* spanning tree algorithm is run on costs
``cost(i, j) = -MI(i, j)``. Minimizing the total cost is exactly equivalent
 to maximizing the sum of mutual information on the selected edges.

Unlike a Hamiltonian chain, the tree is allowed to contain branching nodes.
Every input pair, even one with MI=0, is eligible to connect components.
No scipy sparse-MST zero-edge semantics are imposed on the graph topology.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pycopula_ml.dependence.weight_matrix import CopulaWeightMatrices

from ._validation import _validate_mi_matrix


@dataclass(frozen=True)
class SpanningTreeEdge:
    """Undirected selected edge and its nonnegative MI in nats.

    The edge's optimization cost is ``-mutual_information``.
    """

    source: str
    target: str
    mutual_information: float

    @property
    def cost(self) -> float:
        """Negative MI used by the Kruskal minimum-cost algorithm."""
        return -self.mutual_information


@dataclass(frozen=True)
class SpanningTreeGraph:
    """Maximum-MI tree over all features, represented by selected edges.

    Parameters
    ----------
    feature_names : tuple[str, ...]
        Feature labels in the same order as the input MI matrix.
    edge_indices : tuple[tuple[int, int], ...]
        Undirected edges as zero-based indices with ``i < j``; ordered by
        increasing optimization cost (decreasing mutual information).
    edges : tuple[SpanningTreeEdge, ...]
        Edge labels and MI in the same order as ``edge_indices``.
    total_mi : float
        Sum of the selected MI values in nats. This is *maximized*.
    method : str
        Optimization algorithm, currently ``"kruskal"``.

    Notes
    -----
    The tree has exactly ``d - 1`` edges and is connected and acyclic.
    Unlike a chain, the degree of a node may exceed two.
    """

    feature_names: tuple[str, ...]
    edge_indices: tuple[tuple[int, int], ...]
    edges: tuple[SpanningTreeEdge, ...]
    total_mi: float
    method: str = "kruskal"

    @property
    def total_cost(self) -> float:
        """Return total transformed cost ``-total_mi`` (minimized)."""
        return -self.total_mi

    @property
    def edge_weights(self) -> NDArray[np.float64]:
        """Return nonnegative MI edge weights in selected-edge order."""
        return np.array([edge.mutual_information for edge in self.edges], dtype=float)

    @property
    def edge_costs(self) -> NDArray[np.float64]:
        """Return the negative-MI costs used for selection."""
        return -self.edge_weights

    def adjacency_matrix(self, *, weighted: bool = True) -> NDArray[np.float64]:
        """Return symmetric adjacency in input feature order, no self-loops.

        Use ``weighted=False`` to represent topology even when MI is zero.
        The weighted representation contains zero at every absent edge *and*
        at any selected edge whose MI is exactly zero.
        """
        n = len(self.feature_names)
        adjacency = np.zeros((n, n), dtype=float)
        for (i, j), edge in zip(self.edge_indices, self.edges, strict=True):
            weight = edge.mutual_information if weighted else 1.0
            adjacency[i, j] = adjacency[j, i] = weight
        return adjacency

    def edge_index(self, *, bidirectional: bool = True) -> NDArray[np.int64]:
        """Return ``(2, E)`` indices with optional reverse arcs for GNNs."""
        pairs = list(self.edge_indices)
        if bidirectional:
            pairs.extend((j, i) for i, j in self.edge_indices)
        return np.asarray(pairs, dtype=np.int64).T.copy()

    def edge_attributes(self, *, bidirectional: bool = True) -> NDArray[np.float64]:
        """Return ``(E, 1)`` MI attributes aligned with ``edge_index``."""
        weights = self.edge_weights
        if bidirectional:
            weights = np.tile(weights, 2)
        return weights.reshape(-1, 1)


def build_mst_graph(
    mi_matrix: ArrayLike | CopulaWeightMatrices,
    *,
    feature_names: Sequence[str] | None = None,
) -> SpanningTreeGraph:
    """Build an exact maximum-information spanning tree using Kruskal.

    The algorithm solves a **minimum** spanning tree problem on transformed
    costs ``C_ij = -MI_ij``; equivalently it maximizes total MI among all
    connected, acyclic spanning subgraphs. It does not return a minimum-MI
    tree. No negative MI is required or accepted from the caller.

    Parameters
    ----------
    mi_matrix : array_like or CopulaWeightMatrices
        Nonnegative, finite, symmetric ``(d, d)`` MI matrix in nats, with
        a zero diagonal and at least two features. For CopulaWeightMatrices,
        ``.mutual_information`` is used even when ``weight_method`` selected
        log-likelihood. The input is not modified.
    feature_names : sequence of str, optional
        Unique, nonempty labels. For a CopulaWeightMatrices instance,
        inferred from the object if omitted; otherwise defaults to X1, X2, ...

    Returns
    -------
    SpanningTreeGraph
        Exact optimal weighted spanning tree, with MI weights and the
        corresponding *negative* selection costs accessible independently.

    Raises
    ------
    ValueError
        If matrix dimensions, symmetry, diagonal, MI values, or names fail
        validation.

    Notes
    -----
    All distinct pairs are considered, including zero-MI edges. Equal MI
    edges are resolved deterministically by ascending node indices. The
    algorithm uses ``O(d^2)`` memory and ``O(d^2 log d)`` time for a dense
    matrix of ``d`` features. The MI is assumed to have been estimated
    before calling this function.
    """
    if isinstance(mi_matrix, CopulaWeightMatrices):
        if feature_names is None:
            feature_names = mi_matrix.feature_names
        values = mi_matrix.mutual_information
    else:
        values = mi_matrix
    mi, names = _validate_mi_matrix(values, feature_names)
    n = len(names)

    # Kruskal: sort transformed costs (negative mutual information).
    # The (i, j) entries guarantee deterministic tie breaking.
    cost_edges = sorted(
        (-float(mi[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
    )

    parent = list(range(n))
    size = [1] * n

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    selected: list[tuple[int, int]] = []
    weights: list[float] = []
    for cost, i, j in cost_edges:
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            continue  # This edge would create a cycle.
        if size[root_i] < size[root_j]:
            root_i, root_j = root_j, root_i
        parent[root_j] = root_i
        size[root_i] += size[root_j]
        selected.append((i, j))
        weights.append(-cost)
        if len(selected) == n - 1:
            break

    edges = tuple(
        SpanningTreeEdge(names[i], names[j], weight)
        for (i, j), weight in zip(selected, weights, strict=True)
    )
    return SpanningTreeGraph(
        feature_names=names,
        edge_indices=tuple(selected),
        edges=edges,
        total_mi=float(sum(weights)),
    )
