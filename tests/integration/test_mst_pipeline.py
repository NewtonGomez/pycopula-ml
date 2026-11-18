"""End-to-end rank transformation, Gaussian MLE, MI matrix, and MST."""

import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices
from pycopula_ml.graphs import build_mst_graph


def test_gaussian_copula_mi_to_maximum_spanning_tree():
    rng = np.random.default_rng(2026)
    latent = rng.normal(size=120)
    observations = np.column_stack(
        (
            latent + rng.normal(scale=0.1, size=120),
            rng.normal(size=120),
            0.8 * latent + rng.normal(scale=0.2, size=120),
        )
    )
    pseudo = rankdata(observations, axis=0) / (len(observations) + 1)
    matrices = build_weight_matrices(
        pseudo,
        GaussianCopula,
        bounds=(-0.98, 0.98),
        weight_method="likelihood",  # Verify that the MST reads MI, not LL.
        feature_names=("F3", "F4", "F5"),
    )
    graph = build_mst_graph(matrices)
    assert graph.feature_names == ("F3", "F4", "F5")
    assert len(graph.edges) == 2
    assert graph.total_mi > 0
    assert graph.total_cost == -graph.total_mi
    assert all(edge.mutual_information >= 0 for edge in graph.edges)
    assert np.isfinite(graph.edge_attributes()).all()
    assert graph.edge_index().shape == (2, 4)
    np.testing.assert_array_equal(
        graph.adjacency_matrix(weighted=False).sum(axis=0).sum(), 4
    )
