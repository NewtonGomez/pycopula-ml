"""Test boundary between pairwise MI estimation and graph construction."""

import numpy as np

from pycopula_ml.dependence import CopulaWeightMatrices
from pycopula_ml.graphs import build_chain_graph


def test_copula_weights_wrapper_uses_mi_instead_of_loglikelihood():
    """A likelihood-selected matrix must not silently replace the MI metric."""
    mi = np.array([[0.0, 9.0, 1.0], [9.0, 0.0, 8.0], [1.0, 8.0, 0.0]])
    ll = np.array([[0.0, 1.0, 9.0], [1.0, 0.0, 2.0], [9.0, 2.0, 0.0]])
    matrices = CopulaWeightMatrices(
        log_likelihood=ll,
        theta=np.zeros((3, 3)),
        mutual_information=mi,
        weights=ll,
        feature_names=("a", "b", "c"),
        weight_method="likelihood",
    )
    graph = build_chain_graph(matrices)
    assert graph.permutation == ("a", "b", "c")
    assert graph.total_mi == 17.0
    assert graph.adjacency_matrix()[0, 2] == 0.0
