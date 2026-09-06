# Marginal API

## Public imports

```python
from pycopula_ml.marginals import (
    ContinuousMarginalModel,
    EmpiricalMarginal,
    MarginalCandidate,
    MarginalFitResult,
    MarginalModel,
    MarginalSelector,
    ParametricMarginal,
    bivariate_pseudo_observations,
    pseudo_observations,
)
```

## `MarginalModel`

Abstract interface.

### `fit(values)`

Fit the model.

### `cdf(values)`

Evaluate the cumulative distribution function.

### `transform(values, epsilon=1e-10)`

Compute

$$
u_i=F_X(x_i)
$$

and clip into the open unit interval.

## `ContinuousMarginalModel`

Extends `MarginalModel`.

Requires:

```python
pdf(values)
logpdf(values)
```

## `ParametricMarginal`

Generic wrapper around a continuous SciPy distribution.

Constructor:

```python
ParametricMarginal(
    distribution,
    name=None,
    fit_kwargs=None,
)
```

Important attributes:

```python
parameters_
result_
loc_
scale_
```

Methods:

```python
fit(values)
pdf(values)
logpdf(values)
cdf(values)
transform(values)
support()
```

## `MarginalFitResult`

Structured fit diagnostics.

Fields:

```text
distribution
parameters
sample_size
n_parameters
log_likelihood
aic
bic
ks_statistic
ks_pvalue
```

## `MarginalCandidate`

Defines a distribution considered by `MarginalSelector`.

Example:

```python
from scipy.stats import gamma

candidate = MarginalCandidate(
    name="gamma",
    distribution=gamma,
    fit_kwargs={"floc": 0.0},
)
```

## `MarginalSelector`

Constructor:

```python
MarginalSelector(
    candidates=None,
    criterion="aic",
    small_sample_warning=20,
    boundary_tolerance=1e-6,
)
```

Important attributes:

```python
selected_model_
selected_result_
results_
failed_
warnings_
```

Convenience aliases:

```python
best_model_
best_result_
```

Methods:

```python
fit(values)
transform(values)
fit_transform(values)
ranking()
summary()
```

## `EmpiricalMarginal`

Implements the empirical CDF.

Methods:

```python
fit(values)
cdf(values)
transform(values)
fit_transform(values)
```

## `pseudo_observations(values, ties="average")`

Computes

$$
u_i=\frac{R_i}{n+1}.
$$

## `bivariate_pseudo_observations(x, y, ties="average")`

Returns:

```python
u, v
```

for two paired samples.

The two samples must have the same number of observations.
