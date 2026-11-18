"""Unit tests for mutual-information Hamiltonian path construction."""

import numpy as np
import pytest

from pycopula_ml.graphs import build_chain_graph


@pytest.fixture
def known_chain():
    """Matrix with a unique strong path 0--2--3--1."""
    return np.array(
        [
            [0, 1, 8, 2],
            [1, 0, 3, 9],
            [8, 3, 0, 7],
            [2, 9, 7, 0],
        ],
        dtype=float,
    )


@pytest.mark.parametrize("method", ["greedy", "tsp", "best"])
def test_recovers_known_chain(known_chain, method):
    graph = build_chain_graph(
        known_chain, method=method, feature_names=("A", "B", "C", "D")
    )
    assert graph.order == (0, 2, 3, 1)
    assert graph.permutation == ("A", "C", "D", "B")
    assert len(graph.edges) == 3
    assert graph.total_mi == pytest.approx(24.0)
    assert graph.edge_weights.tolist() == [8.0, 7.0, 9.0]
    np.testing.assert_allclose(graph.adjacency_matrix(), graph.adjacency_matrix().T)
    assert np.count_nonzero(graph.adjacency_matrix()) == 6
    np.testing.assert_array_equal(
        graph.edge_index(bidirectional=False),
        np.array([[0, 2, 3], [2, 3, 1]]),
    )
    assert graph.edge_index().shape == (2, 6)
    np.testing.assert_array_equal(
        graph.edge_attributes().ravel(), [8.0, 7.0, 9.0, 8.0, 7.0, 9.0]
    )
    assert graph.edge_attributes(bidirectional=False).shape == (3, 1)


def test_no_mutation_and_two_nodes():
    mi = np.array([[0.0, 0.5], [0.5, 0.0]])
    original = mi.copy()
    graph = build_chain_graph(mi)
    assert graph.order == (0, 1)
    assert graph.total_mi == pytest.approx(0.5)
    np.testing.assert_array_equal(mi, original)


def test_zero_weights_return_valid_chain():
    graph = build_chain_graph(np.zeros((5, 5)), method="best")
    assert sorted(graph.order) == list(range(5))
    assert graph.total_mi == 0.0
    assert len(graph.edges) == 4
    assert not np.any(graph.adjacency_matrix())
    assert np.count_nonzero(graph.adjacency_matrix(weighted=False)) == 8


def test_best_does_not_underperform_individual_heuristics(known_chain):
    scores = {
        method: build_chain_graph(known_chain, method=method).total_mi
        for method in ("greedy", "tsp", "best")
    }
    assert scores["best"] >= max(scores["greedy"], scores["tsp"])


@pytest.mark.parametrize(
    "matrix,match",
    [
        (np.ones((2, 3)), "square"),
        (np.ones((1, 1)), "square"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
        (np.array([[0.0, 0.2], [0.3, 0.0]]), "symmetric"),
        (np.array([[0.0, -0.2], [-0.2, 0.0]]), "negative"),
        (np.array([[1.0, 0.2], [0.2, 0.0]]), "zero diagonal"),
    ],
)
def test_reject_invalid_matrix(matrix, match):
    with pytest.raises(ValueError, match=match):
        build_chain_graph(matrix)


def test_reject_invalid_labels_and_method(known_chain):
    with pytest.raises(ValueError, match="method"):
        build_chain_graph(known_chain, method="unknown")
    with pytest.raises(ValueError, match="unique"):
        build_chain_graph(known_chain, feature_names=["A"] * 4)
    with pytest.raises(ValueError, match="matching"):
        build_chain_graph(known_chain, feature_names=["A"])


def test_deterministic_ties_and_repeated_calls():
    mi = np.ones((6, 6)) - np.eye(6)
    first = build_chain_graph(mi)
    for _ in range(5):
        assert build_chain_graph(mi) == first
    assert sorted(first.order) == list(range(6))
    assert first.total_mi == pytest.approx(5.0)


def test_path_constraints_for_random_matrices():
    rng = np.random.default_rng(123)
    for n in (3, 6, 10):
        matrix = rng.uniform(size=(n, n))
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        graph = build_chain_graph(matrix)
        assert len(graph.order) == n
        assert len(set(graph.order)) == n
        adj = graph.adjacency_matrix()
        np.testing.assert_allclose(np.diag(adj), 0.0)
        assert graph.total_mi == pytest.approx(
            sum(
                matrix[i, j] for i, j in zip(graph.order, graph.order[1:], strict=False)
            )
        )
