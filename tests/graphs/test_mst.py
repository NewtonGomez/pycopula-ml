"""Unit tests for exact maximum-MI spanning trees via negative-MI Kruskal."""

from itertools import product

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

from pycopula_ml.dependence import CopulaWeightMatrices
from pycopula_ml.graphs import build_chain_graph, build_mst_graph


@pytest.fixture
def example_matrix():
    """Four-variable MI matrix used in the chain-graph documentation."""
    return np.array(
        [[0, 1, 8, 2], [1, 0, 3, 9], [8, 3, 0, 7], [2, 9, 7, 0]],
        dtype=float,
    )


def _check_tree(graph):
    """Verify the tree is connected, acyclic and has valid undirected edges."""
    n = len(graph.feature_names)
    assert len(graph.edges) == len(graph.edge_indices) == n - 1
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in graph.edge_indices:
        assert 0 <= i < j < n
        assert find(i) != find(j)
        parent[find(i)] = find(j)
    assert len({find(i) for i in range(n)}) == 1


def _prufer_score(sequence, weights):
    """Return the weight of the labeled tree defined by a Prüfer sequence."""
    n = len(weights)
    degrees = [1] * n
    for vertex in sequence:
        degrees[vertex] += 1
    total = 0.0
    for vertex in sequence:
        leaf = min(i for i, degree in enumerate(degrees) if degree == 1)
        total += weights[leaf, vertex]
        degrees[leaf] -= 1
        degrees[vertex] -= 1
    a, b = (i for i, degree in enumerate(degrees) if degree == 1)
    return float(total + weights[a, b])


def test_known_matrix_outputs_and_negative_costs(example_matrix):
    graph = build_mst_graph(
        example_matrix, feature_names=["A", "B", "C", "D"]
    )
    assert graph.feature_names == ("A", "B", "C", "D")
    assert graph.method == "kruskal"
    assert graph.edge_indices == ((1, 3), (0, 2), (2, 3))
    assert [(edge.source, edge.target) for edge in graph.edges] == [
        ("B", "D"), ("A", "C"), ("C", "D")
    ]
    assert graph.total_mi == pytest.approx(24.0)
    assert graph.total_cost == pytest.approx(-24.0)
    np.testing.assert_array_equal(graph.edge_weights, [9, 8, 7])
    np.testing.assert_array_equal(graph.edge_costs, [-9, -8, -7])
    assert [edge.cost for edge in graph.edges] == [-9, -8, -7]
    np.testing.assert_array_equal(
        graph.adjacency_matrix(),
        [[0, 0, 8, 0], [0, 0, 0, 9], [8, 0, 0, 7], [0, 9, 7, 0]],
    )
    np.testing.assert_array_equal(
        graph.edge_index(bidirectional=False),
        [[1, 0, 2], [3, 2, 3]],
    )
    np.testing.assert_array_equal(
        graph.edge_index(),
        [[1, 0, 2, 3, 2, 3], [3, 2, 3, 1, 0, 2]],
    )
    np.testing.assert_array_equal(graph.edge_attributes().ravel(), [9, 8, 7] * 2)
    np.testing.assert_array_equal(
        graph.edge_attributes(bidirectional=False).ravel(), [9, 8, 7]
    )
    _check_tree(graph)


def test_tree_allows_branching_not_possible_in_chain():
    mi = np.array(
        [[0, 9, 8, 7], [9, 0, 1, 1], [8, 1, 0, 1], [7, 1, 1, 0]],
        dtype=float,
    )
    tree = build_mst_graph(mi)
    chain = build_chain_graph(mi)
    assert tree.edge_indices == ((0, 1), (0, 2), (0, 3))
    assert tree.total_mi == pytest.approx(24.0)
    assert tree.total_mi > chain.total_mi
    degrees = tree.adjacency_matrix(weighted=False).sum(axis=0)
    np.testing.assert_array_equal(degrees, [3, 1, 1, 1])
    _check_tree(tree)


def test_zero_mi_edges_are_not_dropped():
    graph = build_mst_graph(np.zeros((5, 5)))
    _check_tree(graph)
    assert graph.total_mi == 0.0
    assert graph.total_cost == 0.0
    assert graph.edge_indices == ((0, 1), (0, 2), (0, 3), (0, 4))
    assert np.count_nonzero(graph.adjacency_matrix()) == 0
    assert np.count_nonzero(graph.adjacency_matrix(weighted=False)) == 8
    assert graph.edge_index().shape == (2, 8)
    assert graph.edge_attributes().shape == (8, 1)


def test_two_nodes_returns_one_edge():
    graph = build_mst_graph(np.array([[0, 1.75], [1.75, 0]]))
    assert graph.edge_indices == ((0, 1),)
    assert graph.total_mi == pytest.approx(1.75)
    assert graph.edge_index(bidirectional=False).shape == (2, 1)
    _check_tree(graph)


def test_ties_are_deterministic_and_matrix_is_not_mutated():
    mi = np.ones((6, 6)) - np.eye(6)
    before = mi.copy()
    results = [build_mst_graph(mi) for _ in range(4)]
    assert results == [results[0]] * 4
    assert results[0].edge_indices == tuple((0, i) for i in range(1, 6))
    assert results[0].total_mi == pytest.approx(5.0)
    np.testing.assert_array_equal(mi, before)
    _check_tree(results[0])


def test_wrapper_uses_mutual_information_even_with_likelihood_weights():
    mi = np.array([[0, 9, 1], [9, 0, 8], [1, 8, 0]], dtype=float)
    ll = np.array([[0, 1, 90], [1, 0, 2], [90, 2, 0]], dtype=float)
    matrices = CopulaWeightMatrices(
        log_likelihood=ll,
        theta=np.zeros((3, 3)),
        mutual_information=mi,
        weights=ll,
        feature_names=("F3", "F4", "F5"),
        weight_method="likelihood",
    )
    graph = build_mst_graph(matrices)
    assert graph.feature_names == ("F3", "F4", "F5")
    assert set(graph.edge_indices) == {(0, 1), (1, 2)}
    assert graph.total_mi == pytest.approx(17.0)
    assert build_mst_graph(matrices, feature_names=["a", "b", "c"]).feature_names == (
        "a", "b", "c"
    )


@pytest.mark.parametrize(
    "matrix, match",
    [
        (np.ones((3, 2)), "square"),
        (np.ones((1, 1)), "square"),
        (np.array([[0, float("nan")], [float("nan"), 0]]), "finite"),
        (np.array([[0, float("inf")], [float("inf"), 0]]), "finite"),
        (np.array([[0, 0.2], [0.3, 0]]), "symmetric"),
        (np.array([[0, -0.2], [-0.2, 0]]), "negative"),
        (np.array([[1, 0.2], [0.2, 0]]), "zero diagonal"),
        (["bad", "matrix"], "numeric"),
    ],
)
def test_invalid_matrix_raises_value_error(matrix, match):
    with pytest.raises(ValueError, match=match):
        build_mst_graph(matrix)


@pytest.mark.parametrize(
    "names, match",
    [
        (["a"], "matching"),
        (["a", "a"], "unique"),
        (["a", ""], "nonempty"),
    ],
)
def test_invalid_feature_labels_raises(names, match):
    with pytest.raises(ValueError, match=match):
        build_mst_graph([[0, 1], [1, 0]], feature_names=names)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_matches_global_optimum_enumerating_all_labeled_trees(n):
    rng = np.random.default_rng(900 + n)
    for _ in range(4):
        mi = rng.uniform(0.01, 5, (n, n))
        mi = (mi + mi.T) / 2
        np.fill_diagonal(mi, 0)
        actual = build_mst_graph(mi)
        optimum = max(
            _prufer_score(seq, mi)
            for seq in product(range(n), repeat=n - 2)
        )
        assert actual.total_mi == pytest.approx(optimum, abs=1e-10)
        _check_tree(actual)


def test_matches_scipy_minimum_spanning_tree_on_negative_weights():
    rng = np.random.default_rng(2121)
    for n in (3, 7, 16):
        mi = rng.uniform(0.25, 7.0, size=(n, n))
        mi = (mi + mi.T) / 2
        np.fill_diagonal(mi, 0)
        graph = build_mst_graph(mi)
        scipy_cost = float(minimum_spanning_tree(-mi).sum())
        assert graph.total_cost == pytest.approx(scipy_cost)
        _check_tree(graph)
