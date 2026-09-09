"""Unit tests for automatic marginal distribution selection."""

import numpy as np
import pytest
from scipy.stats import gamma, norm

from pycopula_ml.marginals import (
    MarginalCandidate,
    MarginalSelector,
)


class BrokenDistribution:
    """Dummy distribution that always fails during fitting."""

    name = "broken"

    @staticmethod
    def fit(values, **kwargs):
        """Raise an intentional fitting error."""
        raise ValueError(
            "Intentional fitting failure."
        )


class TestMarginalSelectorInitialization:
    """Tests for MarginalSelector initialization."""

    @pytest.mark.parametrize(
        "criterion",
        [
            "aic",
            "bic",
            "ks",
        ],
    )
    def test_valid_selection_criteria(
        self,
        criterion: str,
    ) -> None:
        """Supported criteria should initialize successfully."""
        selector = MarginalSelector(
            criterion=criterion
        )

        assert selector.criterion == criterion

    def test_invalid_selection_criterion_raises_value_error(
        self,
    ) -> None:
        """Unsupported selection criteria should be rejected."""
        with pytest.raises(ValueError):
            MarginalSelector(
                criterion="invalid"
            )

    def test_selector_is_initially_unfitted(self) -> None:
        """A new selector should not contain a selected model."""
        selector = MarginalSelector()

        assert selector.selected_model_ is None
        assert selector.selected_result_ is None
        assert selector.results_ == []
        assert selector.failed_ == {}


class TestMarginalSelectorFit:
    """Tests for fitting and model selection."""

    def test_fit_selects_a_model(self) -> None:
        """Successful fitting should produce a selected model."""
        x = np.array(
            [
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
            ]
        )

        selector = MarginalSelector(
            candidates=[
                MarginalCandidate(
                    name="normal",
                    distribution=norm,
                ),
            ]
        )

        selector.fit(x)

        assert selector.selected_model_ is not None
        assert selector.selected_result_ is not None

        assert (
            selector.selected_result_.distribution
            == "normal"
        )

    def test_results_are_sorted_by_aic(self) -> None:
        """AIC selection should sort results from lowest to highest AIC."""
        x = np.array(
            [
                0.5,
                0.8,
                1.0,
                1.3,
                1.8,
                2.1,
                2.8,
                3.5,
                4.0,
                5.0,
            ]
        )

        selector = MarginalSelector(
            criterion="aic"
        ).fit(x)

        aic_values = [
            result.aic
            for result in selector.results_
        ]

        assert aic_values == sorted(
            aic_values
        )

        assert selector.selected_result_ is not None

        assert np.isclose(
            selector.selected_result_.aic,
            min(aic_values),
        )

    def test_results_are_sorted_by_bic(self) -> None:
        """BIC selection should sort results from lowest to highest BIC."""
        x = np.array(
            [
                0.5,
                0.8,
                1.0,
                1.3,
                1.8,
                2.1,
                2.8,
                3.5,
                4.0,
                5.0,
            ]
        )

        selector = MarginalSelector(
            criterion="bic"
        ).fit(x)

        bic_values = [
            result.bic
            for result in selector.results_
        ]

        assert bic_values == sorted(
            bic_values
        )

        assert selector.selected_result_ is not None

        assert np.isclose(
            selector.selected_result_.bic,
            min(bic_values),
        )

    def test_results_are_sorted_by_ks(self) -> None:
        """KS selection should minimize the KS statistic."""
        x = np.array(
            [
                0.5,
                0.8,
                1.0,
                1.3,
                1.8,
                2.1,
                2.8,
                3.5,
                4.0,
                5.0,
            ]
        )

        selector = MarginalSelector(
            criterion="ks"
        ).fit(x)

        ks_values = [
            result.ks_statistic
            for result in selector.results_
        ]

        assert ks_values == sorted(
            ks_values
        )

        assert selector.selected_result_ is not None

        assert np.isclose(
            selector.selected_result_.ks_statistic,
            min(ks_values),
        )

    def test_selected_result_matches_first_ranking_result(
        self,
    ) -> None:
        """The selected result should be the first ranked result."""
        x = np.array(
            [
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
            ]
        )

        selector = MarginalSelector().fit(x)

        ranking = selector.ranking()

        assert selector.selected_result_ == ranking[0]

    def test_best_model_alias_matches_selected_model(
        self,
    ) -> None:
        """best_model_ should remain an alias for selected_model_."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        selector = MarginalSelector().fit(x)

        assert (
            selector.best_model_
            is selector.selected_model_
        )

    def test_best_result_alias_matches_selected_result(
        self,
    ) -> None:
        """best_result_ should remain an alias for selected_result_."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        selector = MarginalSelector().fit(x)

        assert (
            selector.best_result_
            is selector.selected_result_
        )


class TestMarginalSelectorTransform:
    """Tests for transformations performed by the selector."""

    def test_transform_returns_open_unit_interval(self) -> None:
        """Selected marginal transformation should produce valid copula data."""
        x = np.array(
            [
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
            ]
        )

        selector = MarginalSelector(
            candidates=[
                MarginalCandidate(
                    name="normal",
                    distribution=norm,
                ),
            ]
        ).fit(x)

        u = selector.transform(x)

        assert np.all(u > 0.0)
        assert np.all(u < 1.0)

    def test_fit_transform_matches_separate_operations(
        self,
    ) -> None:
        """fit_transform should match fit followed by transform."""
        x = np.array(
            [
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
            ]
        )

        candidates = [
            MarginalCandidate(
                name="normal",
                distribution=norm,
            ),
        ]

        selector_a = MarginalSelector(
            candidates=candidates
        )

        selector_a.fit(x)

        expected = selector_a.transform(x)

        selector_b = MarginalSelector(
            candidates=candidates
        )

        result = selector_b.fit_transform(x)

        assert np.allclose(
            result,
            expected,
        )

    def test_transform_before_fit_raises_runtime_error(
        self,
    ) -> None:
        """Transform should require a previously selected model."""
        selector = MarginalSelector()

        with pytest.raises(RuntimeError):
            selector.transform(
                [1.0, 2.0, 3.0]
            )

    def test_ranking_before_fit_raises_runtime_error(
        self,
    ) -> None:
        """Ranking should require a fitted selector."""
        selector = MarginalSelector()

        with pytest.raises(RuntimeError):
            selector.ranking()

    def test_summary_before_fit_raises_runtime_error(
        self,
    ) -> None:
        """Summary should require a fitted selector."""
        selector = MarginalSelector()

        with pytest.raises(RuntimeError):
            selector.summary()


class TestMarginalSelectorDiagnostics:
    """Tests for selector diagnostics and reporting."""

    def test_small_sample_generates_warning(self) -> None:
        """Small samples should generate an explicit diagnostic warning."""
        x = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )

        selector = MarginalSelector(
            candidates=[
                MarginalCandidate(
                    name="normal",
                    distribution=norm,
                ),
            ],
            small_sample_warning=20,
        ).fit(x)

        assert any(
            "sample is small"
            in warning.lower()
            for warning in selector.warnings_
        )

    def test_large_sample_does_not_generate_small_sample_warning(
        self,
    ) -> None:
        """A sample above the configured threshold should avoid the warning."""
        x = np.linspace(
            -3.0,
            3.0,
            100,
        )

        selector = MarginalSelector(
            candidates=[
                MarginalCandidate(
                    name="normal",
                    distribution=norm,
                ),
            ],
            small_sample_warning=20,
        ).fit(x)

        assert not any(
            "sample is small"
            in warning.lower()
            for warning in selector.warnings_
        )

    def test_failed_candidate_is_recorded(self) -> None:
        """A failed distribution should not prevent other candidates fitting."""
        x = np.array(
            [
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
            ]
        )

        selector = MarginalSelector(
            candidates=[
                MarginalCandidate(
                    name="broken",
                    distribution=BrokenDistribution(),
                ),
                MarginalCandidate(
                    name="normal",
                    distribution=norm,
                ),
            ]
        )

        selector.fit(x)

        assert "broken" in selector.failed_

        assert selector.selected_result_ is not None

        assert (
            selector.selected_result_.distribution
            == "normal"
        )

    def test_summary_contains_selection_information(
        self,
    ) -> None:
        """Summary should explain the selected model and criterion."""
        x = np.array(
            [
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
            ]
        )

        selector = MarginalSelector(
            candidates=[
                MarginalCandidate(
                    name="normal",
                    distribution=norm,
                ),
            ],
            criterion="aic",
        ).fit(x)

        summary = selector.summary()

        assert "Marginal distribution selection" in summary
        assert "Criterion: AIC" in summary
        assert "Selected distribution: normal" in summary
        assert "Selection reason" in summary

    def test_fit_kwargs_are_passed_to_distribution(self) -> None:
        """Candidate fitting restrictions should reach the SciPy model."""
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

        selector = MarginalSelector(
            candidates=[
                MarginalCandidate(
                    name="gamma",
                    distribution=gamma,
                    fit_kwargs={
                        "floc": 0.0,
                    },
                ),
            ]
        ).fit(x)

        assert selector.selected_model_ is not None

        assert np.isclose(
            selector.selected_model_.loc_,
            0.0,
        )
