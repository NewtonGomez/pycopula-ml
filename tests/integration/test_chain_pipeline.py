"""Exercise rank transformation, actual Gaussian copula MLE, MI, and graphs."""

import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices
from pycopula_ml.graphs import build_chain_graph


def test_gaussian_copula_to_chain_end_to_end():
    rng = np.random.default_rng(2026)
    base = rng.normal(size=140)
    observations = np.column_stack(
        (
            base + rng.normal(scale=0.1, size=140),
            rng.normal(size=140),
            0.8 * base + rng.normal(scale=0.2, size=140),
        )
    )
    pseudo = rankdata(observations, axis=0) / (len(observations) + 1)
    matrices = build_weight_matrices(
        pseudo,
        GaussianCopula,
        bounds=(-0.98, 0.98),
        weight_method="likelihood",
        feature_names=("F3", "F4", "F5"),
    )
    # The graph explicitly uses MI even though the selected weight method is LL.
    graph = build_chain_graph(matrices)
    assert sorted(graph.permutation) == ["F3", "F4", "F5"]
    assert len(graph.edges) == 2
    assert np.isfinite(graph.total_mi)
    assert graph.total_mi > 0
    assert graph.edge_index().shape == (2, 4)
    assert graph.edge_attributes().shape == (4, 1)
    assert graph.adjacency_matrix().shape == (3, 3)
