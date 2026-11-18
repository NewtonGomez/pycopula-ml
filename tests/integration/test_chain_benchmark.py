"""Tests for reproducible benchmark inputs and small exact-solver oracle."""

from itertools import permutations

import numpy as np
import pytest

from scripts.benchmark_chain_algorithms import (
    benchmark,
    exact_maximum_path_score,
    gaussian_mi_matrix,
    summarize,
)


@pytest.mark.parametrize("regime", ["ar1", "blocks", "factors"])
def test_gaussian_mi_generator_is_reproducible_and_symmetric(regime):
    a = gaussian_mi_matrix(6, regime, seed=7)
    b = gaussian_mi_matrix(6, regime, seed=7)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, a.T)
    np.testing.assert_array_equal(np.diag(a), 0)
    assert np.isfinite(a).all()
    assert (a >= 0).all()


def test_exact_solver_matches_brute_force():
    weights = gaussian_mi_matrix(6, "factors", seed=3)
    exact = exact_maximum_path_score(weights)
    brute = max(
        sum(weights[a, b] for a, b in zip(path[:-1], path[1:], strict=True))
        for path in permutations(range(6))
    )
    assert exact == pytest.approx(brute)


def test_benchmark_produces_paired_rows_and_summary():
    rows = benchmark((4,), (1,), ("ar1",), repeats=1)
    assert len(rows) == 5
    stats = summarize(rows)
    assert len(stats) == 5
    for item in stats:
        assert item["cases"] == 1
        assert item["median_wall_ms"] >= 0
        assert item["median_cpu_ms"] >= 0
        assert item["median_traced_peak_kib"] > 0
        assert item["mean_exact_gap_pct"] >= -1e-8
