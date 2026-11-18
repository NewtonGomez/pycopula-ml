"""Unit tests for Akaike and Bayesian information criteria."""

import math

import numpy as np
import pytest

from pycopula_ml.model_selection import aic, bic


def test_aic_bic_closed_forms():
    assert aic(10.0, 2) == pytest.approx(-16.0)
    assert bic(10.0, 2, 100) == pytest.approx(2 * math.log(100) - 20.0)
    assert aic(10.0, 0) == -20.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reject_invalid_log_likelihood(value):
    with pytest.raises(ValueError, match="finite"):
        aic(value, 1)
    with pytest.raises(ValueError, match="finite"):
        bic(value, 1, 10)


@pytest.mark.parametrize("value", [True, 0.5, -1, "a"])
def test_reject_invalid_parameter_counts(value):
    with pytest.raises((TypeError, ValueError)):
        aic(0.0, value)
    with pytest.raises((TypeError, ValueError)):
        bic(0.0, value, 10)


@pytest.mark.parametrize("value", [True, 0, -1, 2.5, np.inf])
def test_reject_invalid_observation_counts(value):
    with pytest.raises((TypeError, ValueError)):
        bic(0.0, 1, value)
