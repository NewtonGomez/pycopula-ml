"""Unit tests for the bivariate Frank copula."""

import numpy as np
import pytest

from pycopula_ml.copulas.frank import FrankCopula


class TestFrankCopulaInitialization:
    """Tests for Frank copula initialization and parameter validation."""

    def test_valid_positive_theta(self) -> None:
        """A positive finite theta should initialize correctly."""
        copula = FrankCopula(theta=2.0)

        assert copula.theta == 2.0

    def test_valid_negative_theta(self) -> None:
        """A negative finite theta should initialize correctly."""
        copula = FrankCopula(theta=-2.0)

        assert copula.theta == -2.0

    def test_theta_zero_raises_value_error(self) -> None:
        """Theta equal to zero should raise a ValueError."""
        with pytest.raises(ValueError):
            FrankCopula(theta=0.0)

    @pytest.mark.parametrize(
        "theta",
        [np.inf, -np.inf, np.nan],
    )
    def test_non_finite_theta_raises_value_error(
        self,
        theta: float,
    ) -> None:
        """Non-finite theta values should raise a ValueError."""
        with pytest.raises(ValueError):
            FrankCopula(theta=theta)


class TestFrankCopulaCDF:
    """Tests for the Frank copula cumulative distribution function."""

    def test_cdf_known_value(self) -> None:
        """CDF should match a known reference value."""
        copula = FrankCopula(theta=2.0)

        result = copula.cdf(0.3, 0.7)

        expected = 0.24972133337304844

        assert np.isclose(result, expected, rtol=1e-12)

    def test_cdf_lower_boundary_u(self) -> None:
        """The copula should satisfy C(0, v) = 0."""
        copula = FrankCopula(theta=2.0)
        v = np.linspace(0.0, 1.0, 11)

        result = copula.cdf(0.0, v)

        assert np.allclose(result, 0.0)

    def test_cdf_lower_boundary_v(self) -> None:
        """The copula should satisfy C(u, 0) = 0."""
        copula = FrankCopula(theta=2.0)
        u = np.linspace(0.0, 1.0, 11)

        result = copula.cdf(u, 0.0)

        assert np.allclose(result, 0.0)

    def test_cdf_upper_boundary_u(self) -> None:
        """The copula should satisfy C(1, v) = v."""
        copula = FrankCopula(theta=2.0)
        v = np.linspace(0.0, 1.0, 11)

        result = copula.cdf(1.0, v)

        assert np.allclose(result, v)

    def test_cdf_upper_boundary_v(self) -> None:
        """The copula should satisfy C(u, 1) = u."""
        copula = FrankCopula(theta=2.0)
        u = np.linspace(0.0, 1.0, 11)

        result = copula.cdf(u, 1.0)

        assert np.allclose(result, u)

    def test_cdf_is_symmetric(self) -> None:
        """The Frank copula should satisfy C(u, v) = C(v, u)."""
        copula = FrankCopula(theta=2.0)

        u = np.array([0.1, 0.3, 0.8])
        v = np.array([0.7, 0.5, 0.2])

        assert np.allclose(
            copula.cdf(u, v),
            copula.cdf(v, u),
        )

    def test_cdf_values_are_probabilities(self) -> None:
        """CDF values should remain inside the interval [0, 1]."""
        copula = FrankCopula(theta=2.0)

        u = np.linspace(0.0, 1.0, 20)
        v = np.linspace(0.0, 1.0, 20)

        u_grid, v_grid = np.meshgrid(u, v)

        result = copula.cdf(u_grid, v_grid)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestFrankCopulaPDF:
    """Tests for the Frank copula probability density function."""

    def test_pdf_known_value(self) -> None:
        """PDF should match a known reference value."""
        copula = FrankCopula(theta=2.0)

        result = copula.pdf(0.3, 0.7)

        expected = 0.8499701667295133

        assert np.isclose(result, expected, rtol=1e-12)

    def test_pdf_is_positive(self) -> None:
        """Frank copula density should be positive inside the unit square."""
        copula = FrankCopula(theta=2.0)

        u = np.linspace(0.01, 0.99, 20)
        v = np.linspace(0.01, 0.99, 20)

        u_grid, v_grid = np.meshgrid(u, v)

        result = copula.pdf(u_grid, v_grid)

        assert np.all(result > 0.0)

    def test_pdf_is_symmetric(self) -> None:
        """The Frank density should satisfy c(u, v) = c(v, u)."""
        copula = FrankCopula(theta=2.0)

        u = np.array([0.2, 0.4, 0.8])
        v = np.array([0.7, 0.5, 0.3])

        assert np.allclose(
            copula.pdf(u, v),
            copula.pdf(v, u),
        )


class TestFrankCopulaLogPDF:
    """Tests for the logarithm of the Frank copula density."""

    def test_logpdf_known_value(self) -> None:
        """Log-PDF should match a known reference value."""
        copula = FrankCopula(theta=2.0)

        result = copula.logpdf(0.3, 0.7)

        expected = -0.16255402807900138

        assert np.isclose(result, expected, rtol=1e-12)

    def test_logpdf_matches_log_of_pdf(self) -> None:
        """Direct log-PDF should agree with log(PDF)."""
        copula = FrankCopula(theta=2.0)

        u = np.array([0.2, 0.4, 0.7])
        v = np.array([0.3, 0.6, 0.8])

        expected = np.log(copula.pdf(u, v))
        result = copula.logpdf(u, v)

        assert np.allclose(result, expected)


class TestFrankCopulaLikelihood:
    """Tests for the generic copula log-likelihood."""

    def test_log_likelihood_equals_sum_of_logpdf(self) -> None:
        """Log-likelihood should equal the sum of sample log-densities."""
        copula = FrankCopula(theta=2.0)

        u = np.array([0.2, 0.4, 0.7, 0.8])
        v = np.array([0.3, 0.6, 0.5, 0.9])

        expected = np.sum(copula.logpdf(u, v))
        result = copula.log_likelihood(u, v)

        assert np.isclose(result, expected)


class TestFrankCopulaInputValidation:
    """Tests for validation of copula input values."""

    @pytest.mark.parametrize(
        "u",
        [-0.1, 1.1],
    )
    def test_invalid_u_raises_value_error(
        self,
        u: float,
    ) -> None:
        """Values of u outside [0, 1] should raise ValueError."""
        copula = FrankCopula(theta=2.0)

        with pytest.raises(ValueError):
            copula.cdf(u, 0.5)

    @pytest.mark.parametrize(
        "v",
        [-0.1, 1.1],
    )
    def test_invalid_v_raises_value_error(
        self,
        v: float,
    ) -> None:
        """Values of v outside [0, 1] should raise ValueError."""
        copula = FrankCopula(theta=2.0)

        with pytest.raises(ValueError):
            copula.cdf(0.5, v)

    def test_nan_input_raises_value_error(self) -> None:
        """Non-finite marginal values should raise ValueError."""
        copula = FrankCopula(theta=2.0)

        with pytest.raises(ValueError):
            copula.cdf(np.nan, 0.5)
