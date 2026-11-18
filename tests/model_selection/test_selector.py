"""Unit and integration tests for model selection across copula families."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import rankdata

from pycopula_ml.copulas import FrankCopula, GaussianCopula
from pycopula_ml.model_selection import copula_selector

selector_module = import_module("pycopula_ml.model_selection.copula_selector")
CANDIDATES = {FrankCopula: (-5.0, 5.0), GaussianCopula: (-0.9, 0.9)}


def sample():
    u = np.array([0.1, 0.2, 0.5, 0.8])
    v = np.array([0.2, 0.4, 0.6, 0.9])
    return u, v


@pytest.mark.parametrize("criterion", ["aic", "bic", "log_likelihood"])
def test_ranks_candidates_and_exposes_metadata(monkeypatch, criterion, capsys):
    def fake_fit(copula, u, v, bounds):
        assert len(u) == len(v) == 4
        assert bounds == CANDIDATES[copula]
        return SimpleNamespace(
            theta=0.5,
            log_likelihood=5.0 if copula is GaussianCopula else 2.0,
        )

    monkeypatch.setattr(selector_module, "fit_copula_mle", fake_fit)
    u, v = sample()
    result = copula_selector(CANDIDATES, u, v, criterion=criterion, print_results=True)
    assert result.selected.name == "GaussianCopula"
    assert result.candidates[1].name == "FrankCopula"
    assert result.criterion == criterion
    assert result.candidates[0].theta == 0.5
    assert result.candidates[0].aic == pytest.approx(-8.0)
    assert "Selected copula: GaussianCopula" in capsys.readouterr().out


def test_reject_invalid_criterion(monkeypatch):
    monkeypatch.setattr(
        selector_module,
        "fit_copula_mle",
        lambda *_args, **_kwargs: SimpleNamespace(theta=0.0, log_likelihood=1.0),
    )
    u, v = sample()
    with pytest.raises(ValueError, match="criterion"):
        copula_selector(CANDIDATES, u, v, criterion="invalid")


@pytest.mark.parametrize(
    "u,v,fragment",
    [
        ([0.2], [0.2, 0.5], "same"),
        ([], [], "at least one"),
        ([[0.2], [0.3]], [[0.4], [0.6]], "one-dimensional"),
        ([np.nan, 0.3], [0.2, 0.5], "finite"),
    ],
)
def test_reject_invalid_paired_data(u, v, fragment):
    with pytest.raises(ValueError, match=fragment):
        copula_selector(CANDIDATES, u, v)


def test_reject_empty_candidates():
    u, v = sample()
    with pytest.raises(ValueError, match="At least one"):
        copula_selector({}, u, v)


@pytest.mark.parametrize(
    "candidates,error",
    [
        ({object: (0, 1)}, TypeError),
        ({GaussianCopula: (-1, 0, 1)}, ValueError),
        ({GaussianCopula: (0.2, -0.2)}, ValueError),
        ({GaussianCopula: (np.nan, 1.0)}, ValueError),
    ],
)
def test_reject_invalid_candidates(candidates, error):
    u, v = sample()
    with pytest.raises(error):
        copula_selector(candidates, u, v)


def test_real_fit_recovers_finite_scores():
    rng = np.random.default_rng(21)
    x = rng.normal(size=100)
    y = 0.7 * x + rng.normal(scale=0.7, size=100)
    u = rankdata(x) / 101.0
    v = rankdata(y) / 101.0
    result = copula_selector(
        {GaussianCopula: (-0.95, 0.95), FrankCopula: (-8.0, 8.0)},
        u,
        v,
        criterion="bic",
    )
    assert len(result.candidates) == 2
    assert all(np.isfinite(candidate.bic) for candidate in result.candidates)
    assert np.isfinite(result.selected.theta)
