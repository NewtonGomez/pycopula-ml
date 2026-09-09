"""Unit tests for parametric marginal distribution models."""

import numpy as np
import pytest
from scipy.stats import gamma, norm

from pycopula_ml.marginals import ParametricMarginal


class TestParametricMarginalInitialization:
    """Tests for ParametricMarginal initialization."""

    def test_distribution_name_is_detected(self) -> None:
        """The SciPy distribution name should be used by default."""
        marginal = ParametricMarginal(
            norm
        )

        assert marginal.name == "norm"

    def test_custom_distribution_name_is_used(self) -> None:
        """A custom human-readable name should override the SciPy name."""
        marginal = ParametricMarginal(
            norm,
            name="normal",
        )

        assert marginal.name == "normal"

    def test_model_is_initially_unfitted(self) -> None:
        """A new marginal model should not contain fitted parameters."""
        marginal = ParametricMarginal(
            norm
        )

        assert marginal.parameters_ is None
        assert marginal.result_ is None


class TestNormalMarginal:
    """Tests using a Normal SciPy marginal distribution."""

    def test_normal_fit_estimates_mean_and_scale(self) -> None:
        """Normal fit should reproduce sample mean and population std."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm,
            name="normal",
        )

        marginal.fit(x)

        expected_mean = np.mean(x)
        expected_scale = np.std(
            x,
            ddof=0,
        )

        assert marginal.parameters_ is not None

        loc, scale = marginal.parameters_

        assert np.isclose(
            loc,
            expected_mean,
        )

        assert np.isclose(
            scale,
            expected_scale,
        )

    def test_result_contains_sample_size(self) -> None:
        """The fit result should store the fitted sample size."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm,
            name="normal",
        ).fit(x)

        assert marginal.result_ is not None
        assert marginal.result_.sample_size == 5

    def test_normal_has_two_free_parameters(self) -> None:
        """A freely fitted Normal distribution has loc and scale."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm,
            name="normal",
        ).fit(x)

        assert marginal.result_ is not None
        assert marginal.result_.n_parameters == 2

    def test_fit_statistics_are_finite(self) -> None:
        """Log-likelihood, AIC, BIC, and KS should be finite."""
        x = np.array(
            [
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
            ]
        )

        marginal = ParametricMarginal(
            norm,
            name="normal",
        ).fit(x)

        result = marginal.result_

        assert result is not None

        assert np.isfinite(
            result.log_likelihood
        )

        assert np.isfinite(
            result.aic
        )

        assert np.isfinite(
            result.bic
        )

        assert np.isfinite(
            result.ks_statistic
        )

        assert np.isfinite(
            result.ks_pvalue
        )

    def test_pdf_is_positive(self) -> None:
        """Normal fitted density should be positive at finite values."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        values = np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )

        density = marginal.pdf(
            values
        )

        assert np.all(
            density > 0.0
        )

    def test_logpdf_matches_log_of_pdf(self) -> None:
        """Direct log-PDF should equal the logarithm of the PDF."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        values = np.array(
            [
                2.0,
                3.0,
                4.0,
            ]
        )

        expected = np.log(
            marginal.pdf(values)
        )

        result = marginal.logpdf(
            values
        )

        assert np.allclose(
            result,
            expected,
        )

    def test_cdf_is_between_zero_and_one(self) -> None:
        """A fitted CDF should return valid cumulative probabilities."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        probabilities = marginal.cdf(
            x
        )

        assert np.all(
            probabilities >= 0.0
        )

        assert np.all(
            probabilities <= 1.0
        )

    def test_cdf_is_non_decreasing(self) -> None:
        """The fitted CDF should be monotonic."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        values = np.linspace(
            -5.0,
            10.0,
            100,
        )

        probabilities = marginal.cdf(
            values
        )

        assert np.all(
            np.diff(probabilities) >= 0.0
        )

    def test_transform_returns_open_unit_interval(self) -> None:
        """Probability integral transform should avoid zero and one."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        u = marginal.transform(x)

        assert np.all(u > 0.0)
        assert np.all(u < 1.0)

    def test_loc_property_matches_fitted_parameter(self) -> None:
        """loc_ should expose the fitted location parameter."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        assert np.isclose(
            marginal.loc_,
            np.mean(x),
        )

    def test_scale_property_matches_fitted_parameter(self) -> None:
        """scale_ should expose the fitted scale parameter."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        assert np.isclose(
            marginal.scale_,
            np.std(x, ddof=0),
        )

    def test_normal_support_is_unbounded(self) -> None:
        """The Normal distribution should have infinite support."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        lower, upper = marginal.support()

        assert np.isneginf(lower)
        assert np.isposinf(upper)


class TestGammaMarginal:
    """Tests using a Gamma SciPy marginal distribution."""

    def test_fixed_location_is_respected(self) -> None:
        """Gamma fitting with floc=0 should keep location at zero."""
        x = np.array(
            [
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
            ]
        )

        marginal = ParametricMarginal(
            gamma,
            name="gamma",
            fit_kwargs={
                "floc": 0.0,
            },
        ).fit(x)

        assert np.isclose(
            marginal.loc_,
            0.0,
        )

    def test_fixed_location_reduces_free_parameter_count(
        self,
    ) -> None:
        """Gamma with fixed loc should estimate shape and scale only."""
        x = np.array(
            [
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
            ]
        )

        marginal = ParametricMarginal(
            gamma,
            name="gamma",
            fit_kwargs={
                "floc": 0.0,
            },
        ).fit(x)

        assert marginal.result_ is not None

        assert (
            marginal.result_.n_parameters
            == 2
        )

    def test_gamma_support_starts_at_zero_with_fixed_loc(
        self,
    ) -> None:
        """Gamma support should start at zero when loc is fixed to zero."""
        x = np.array(
            [
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
            ]
        )

        marginal = ParametricMarginal(
            gamma,
            fit_kwargs={
                "floc": 0.0,
            },
        ).fit(x)

        lower, upper = marginal.support()

        assert np.isclose(
            lower,
            0.0,
        )

        assert np.isposinf(
            upper
        )


class TestParametricMarginalValidation:
    """Tests for invalid ParametricMarginal operations."""

    def test_pdf_before_fit_raises_runtime_error(self) -> None:
        """PDF evaluation should require a fitted model."""
        marginal = ParametricMarginal(
            norm
        )

        with pytest.raises(RuntimeError):
            marginal.pdf([1.0])

    def test_logpdf_before_fit_raises_runtime_error(self) -> None:
        """Log-PDF evaluation should require a fitted model."""
        marginal = ParametricMarginal(
            norm
        )

        with pytest.raises(RuntimeError):
            marginal.logpdf([1.0])

    def test_cdf_before_fit_raises_runtime_error(self) -> None:
        """CDF evaluation should require a fitted model."""
        marginal = ParametricMarginal(
            norm
        )

        with pytest.raises(RuntimeError):
            marginal.cdf([1.0])

    def test_fit_rejects_empty_sample(self) -> None:
        """An empty sample should not be fitted."""
        marginal = ParametricMarginal(
            norm
        )

        with pytest.raises(ValueError):
            marginal.fit([])

    def test_fit_rejects_multidimensional_sample(self) -> None:
        """Marginal fitting requires one-dimensional data."""
        marginal = ParametricMarginal(
            norm
        )

        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )

        with pytest.raises(ValueError):
            marginal.fit(x)

    def test_fit_rejects_nan(self) -> None:
        """Non-finite observations should be rejected."""
        marginal = ParametricMarginal(
            norm
        )

        x = np.array(
            [
                1.0,
                np.nan,
                3.0,
            ]
        )

        with pytest.raises(ValueError):
            marginal.fit(x)

    @pytest.mark.parametrize(
        "epsilon",
        [
            0.0,
            -0.1,
            0.5,
            1.0,
        ],
    )
    def test_invalid_transform_epsilon_raises_value_error(
        self,
        epsilon: float,
    ) -> None:
        """Transform clipping epsilon must lie between zero and 0.5."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        )

        marginal = ParametricMarginal(
            norm
        ).fit(x)

        with pytest.raises(ValueError):
            marginal.transform(
                x,
                epsilon=epsilon,
            )
