# Estimation testing and statistical validation

Parameter estimation requires more than one kind of test.

A deterministic software test and a Monte Carlo validation experiment answer
different questions and should not be mixed.

## Unit tests

The estimation unit tests verify the behavior of the optimization code itself.

A useful deterministic test defines a synthetic log-likelihood with a known
maximum, for example

$$
\ell(\theta)
=
-(\theta-2.5)^2.
$$

Its exact optimum is

$$
\widehat{\theta}=2.5.
$$

The optimizer can therefore be tested without depending on the Frank
implementation.

Current estimation unit tests should cover:

- recovery of a known deterministic maximum;
- selection of the best result across disjoint intervals;
- rejection of values outside \((0,1)\);
- rejection of mismatched sample shapes;
- rejection of reversed parameter bounds.

Run them with:

```bash
python -m pytest tests/estimation -v
```

These tests belong in `tests/` because they are deterministic, fast, and
suitable for continuous integration.

## Integration tests

An integration test answers a broader question:

> Do the marginal transformation, copula implementation, and estimation layer
> work correctly together?

A representative integration path is:

```text
x, y
  ↓
bivariate_pseudo_observations()
  ↓
u, v
  ↓
FrankCopula
  ↓
fit_copula_mle()
  ↓
theta_hat
```

Integration tests should use fixed data and deterministic expected behavior so
they remain stable in CI.

At the current stage, this is the recommended next testing layer after the
estimation unit tests.

## Statistical validation

A Monte Carlo experiment answers a statistical question rather than a software
unit-test question.

For a known generating parameter,

$$
\theta_{\text{true}},
$$

generate repeated samples

$$
(U_i,V_i)
\sim
C_{\text{Frank}}(\theta_{\text{true}})
$$

and estimate

$$
\widehat{\theta}
$$

for each sample.

The validation script can then study the sampling behavior of the estimator.

This belongs under a path such as:

```text
scripts/validation/
```

rather than the core unit-test suite.

## Bias

For \(R\) repeated experiments,

$$
\operatorname{Bias}
=
\frac{1}{R}
\sum_{r=1}^{R}
(\widehat{\theta}_r-\theta_{\text{true}}).
$$

Bias measures systematic directional error.

- positive bias: tendency to overestimate;
- negative bias: tendency to underestimate;
- bias near zero: no strong systematic direction in the experiment.

## MAE

The mean absolute error is

$$
MAE
=
\frac{1}{R}
\sum_{r=1}^{R}
\left|
\widehat{\theta}_r-\theta_{\text{true}}
\right|.
$$

It measures the average magnitude of the estimation error.

Unlike bias, positive and negative errors do not cancel.

## RMSE

The root mean squared error is

$$
RMSE
=
\sqrt{
\frac{1}{R}
\sum_{r=1}^{R}
(\widehat{\theta}_r-\theta_{\text{true}})^2
}.
$$

RMSE gives greater weight to unusually large errors than MAE.

Typically,

$$
RMSE\geq MAE.
$$

## Recommended validation grid

A stronger statistical validation should vary both dependence strength and
sample size.

For example:

```text
theta_true = 1, 2, 5, 10
n          = 25, 50, 100, 250, 500, 1000
```

For each configuration, repeat the experiment many times and record:

- mean \(\widehat{\theta}\);
- standard deviation of \(\widehat{\theta}\);
- bias;
- MAE;
- RMSE;
- optimizer success rate.

The expected large-sample behavior is that estimation error decreases as the
sample size grows.

This is evidence about the statistical estimator. It is not a replacement for
deterministic unit tests.
