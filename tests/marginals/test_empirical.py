"""Unit tests for empirical marginal distribution utilities."""

import numpy as np
import pytest

from pycopula_ml.marginals import (
    EmpiricalMarginal,
    bivariate_pseudo_observations,
    pseudo_observations,
)


class TestEmpiricalMarginal:
    """Tests for the EmpiricalMarginal class."""

    def test_fit_stores_sample_information(self) -> None:
        """Fitting should store the sample and its sorted representation."""
        x = np.array([4.0, 1.0, 3.0, 2.0])

        marginal = EmpiricalMarginal()
        marginal.fit(x)

        assert marginal.n_samples_ == 4

        assert np.array_equal(
            marginal.sample_,
            x,
        )

        assert np.array_equal(
            marginal.sorted_sample_,
            np.array([1.0, 2.0, 3.0, 4.0]),
        )

    def test_cdf_before_fit_raises_runtime_error(self) -> None:
        """CDF evaluation should fail when the model is not fitted."""
        marginal = EmpiricalMarginal()

        with pytest.raises(RuntimeError):
            marginal.cdf([1.0, 2.0])

    def test_cdf_matches_known_values(self) -> None:
        """The ECDF should equal the observed cumulative proportions."""
        x = np.array(
            [
                10.0,
                20.0,
                30.0,
                40.0,
            ]
        )

        marginal = EmpiricalMarginal().fit(x)

        values = np.array(
            [
                5.0,
                10.0,
                25.0,
                40.0,
                50.0,
            ]
        )

        result = marginal.cdf(values)

        expected = np.array(
            [
                0.0,
                0.25,
                0.50,
                1.0,
                1.0,
            ]
        )

        assert np.allclose(
            result,
            expected,
        )

    def test_cdf_handles_tied_observations(self) -> None:
        """The ECDF should include all observations equal to x."""
        x = np.array(
            [
                1.0,
                2.0,
                2.0,
                4.0,
            ]
        )

        marginal = EmpiricalMarginal().fit(x)

        result = marginal.cdf(2.0)

        assert np.isclose(
            result,
            0.75,
        )

    def test_cdf_is_non_decreasing(self) -> None:
        """The empirical CDF should never decrease."""
        x = np.array(
            [
                3.0,
                1.0,
                5.0,
                2.0,
                4.0,
            ]
        )

        marginal = EmpiricalMarginal().fit(x)

        evaluation_values = np.linspace(
            0.0,
            6.0,
            100,
        )

        probabilities = marginal.cdf(
            evaluation_values
        )

        differences = np.diff(
            probabilities
        )

        assert np.all(
            differences >= 0.0
        )

    def test_fit_transform_returns_open_interval(self) -> None:
        """ECDF transformation should avoid exact zero and one."""
        x = np.array(
            [
                10.0,
                20.0,
                30.0,
                40.0,
            ]
        )

        marginal = EmpiricalMarginal()

        u = marginal.fit_transform(x)

        assert np.all(u > 0.0)
        assert np.all(u < 1.0)

    def test_fit_rejects_multidimensional_sample(self) -> None:
        """Fitting should reject multidimensional observations."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )

        marginal = EmpiricalMarginal()

        with pytest.raises(ValueError):
            marginal.fit(x)

    def test_fit_rejects_empty_sample(self) -> None:
        """Fitting should reject an empty sample."""
        marginal = EmpiricalMarginal()

        with pytest.raises(ValueError):
            marginal.fit([])

    def test_fit_rejects_non_finite_values(self) -> None:
        """Fitting should reject NaN and infinite observations."""
        x = np.array(
            [
                1.0,
                np.nan,
                3.0,
            ]
        )

        marginal = EmpiricalMarginal()

        with pytest.raises(ValueError):
            marginal.fit(x)


class TestPseudoObservations:
    """Tests for rank-based pseudo-observation transformations."""

    def test_known_pseudo_observations(self) -> None:
        """Pseudo-observations should match rank divided by n + 1."""
        x = np.array(
            [
                40.0,
                10.0,
                30.0,
                20.0,
            ]
        )

        result = pseudo_observations(x)

        expected = np.array(
            [
                0.8,
                0.2,
                0.6,
                0.4,
            ]
        )

        assert np.allclose(
            result,
            expected,
        )

    def test_pseudo_observations_are_inside_open_interval(
        self,
    ) -> None:
        """Pseudo-observations should be strictly between zero and one."""
        x = np.array(
            [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ]
        )

        u = pseudo_observations(x)

        assert np.all(u > 0.0)
        assert np.all(u < 1.0)

    def test_pseudo_observations_preserve_order(self) -> None:
        """Larger observations should receive larger ranks."""
        x = np.array(
            [
                8.0,
                2.0,
                5.0,
                1.0,
            ]
        )

        u = pseudo_observations(x)

        assert np.array_equal(
            np.argsort(x),
            np.argsort(u),
        )

    def test_average_ranks_are_used_for_ties(self) -> None:
        """Tied observations should receive their average rank."""
        x = np.array(
            [
                10.0,
                20.0,
                20.0,
                40.0,
            ]
        )

        u = pseudo_observations(
            x,
            ties="average",
        )

        expected = np.array(
            [
                0.2,
                0.5,
                0.5,
                0.8,
            ]
        )

        assert np.allclose(
            u,
            expected,
        )

    def test_tied_values_receive_equal_pseudo_observations(
        self,
    ) -> None:
        """Equal observations should receive equal average ranks."""
        x = np.array(
            [
                1.0,
                2.0,
                2.0,
                4.0,
            ]
        )

        u = pseudo_observations(x)

        assert np.isclose(
            u[1],
            u[2],
        )

    def test_invalid_sample_raises_value_error(self) -> None:
        """Pseudo-observation transformation should validate input."""
        x = np.array(
            [
                1.0,
                np.nan,
                3.0,
            ]
        )

        with pytest.raises(ValueError):
            pseudo_observations(x)


class TestBivariatePseudoObservations:
    """Tests for bivariate rank transformations."""

    def test_known_bivariate_pseudo_observations(self) -> None:
        """Both variables should be transformed independently by rank."""
        x = np.array(
            [
                40.0,
                10.0,
                30.0,
                20.0,
            ]
        )

        y = np.array(
            [
                100.0,
                130.0,
                110.0,
                120.0,
            ]
        )

        u, v = bivariate_pseudo_observations(
            x,
            y,
        )

        expected_u = np.array(
            [
                0.8,
                0.2,
                0.6,
                0.4,
            ]
        )

        expected_v = np.array(
            [
                0.2,
                0.8,
                0.4,
                0.6,
            ]
        )

        assert np.allclose(
            u,
            expected_u,
        )

        assert np.allclose(
            v,
            expected_v,
        )

    def test_bivariate_outputs_have_same_size_as_input(
        self,
    ) -> None:
        """The transformation should preserve sample size."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        )

        y = np.array(
            [
                5.0,
                4.0,
                6.0,
            ]
        )

        u, v = bivariate_pseudo_observations(
            x,
            y,
        )

        assert u.shape == x.shape
        assert v.shape == y.shape

    def test_different_sample_sizes_raise_value_error(
        self,
    ) -> None:
        """Paired variables must contain the same number of observations."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        )

        y = np.array(
            [
                1.0,
                2.0,
            ]
        )

        with pytest.raises(ValueError):
            bivariate_pseudo_observations(
                x,
                y,
            )
