# Estimation API

## Public imports

```python
from pycopula_ml.estimation import (
    CopulaFitResult,
    fit_copula_mle,
)
```

## `fit_copula_mle`

Estimate a scalar copula parameter by maximizing the sample copula
log-likelihood.

Conceptual signature:

```python
fit_copula_mle(
    copula,
    u,
    v,
    bounds,
    *,
    xatol=1e-8,
    maxiter=500,
) -> CopulaFitResult
```

### Parameters

#### `copula`

Copula class or factory accepting

```python
theta=<float>
```

and returning an object that implements

```python
log_likelihood(u, v)
```

Example:

```python
FrankCopula
```

Pass the class, not an already instantiated object, because the optimizer must
construct candidate copulas for many values of `theta`.

#### `u`, `v`

Paired pseudo-observations.

Requirements:

```text
one-dimensional
same shape
non-empty
finite
strictly inside (0, 1)
```

#### `bounds`

Either one interval:

```python
bounds=(1e-6, 50.0)
```

or several intervals:

```python
bounds=(
    (-50.0, -1e-6),
    (1e-6, 50.0),
)
```

Every interval must satisfy

$$
\text{lower}<\text{upper}.
$$

#### `xatol`

Absolute scalar-parameter tolerance used by the bounded numerical optimizer.

Default:

```text
1e-8
```

#### `maxiter`

Maximum iterations per interval.

Default:

```text
500
```

### Returns

A `CopulaFitResult`.

### Raises

`ValueError` for invalid pseudo-observations, bounds, `xatol`, or `maxiter`.

`RuntimeError` if all searched intervals fail to produce a finite successful
optimization.

## `CopulaFitResult`

Structured result returned by `fit_copula_mle()`.

Fields:

```text
theta
log_likelihood
success
message
nfev
n_obs
interval
raw_result
interval_results
```

### `theta`

Estimated dependence parameter.

### `log_likelihood`

Maximized copula log-likelihood:

$$
\ell(\widehat{\theta}).
$$

### `success`

Whether the selected SciPy optimization converged successfully.

### `message`

Optimizer convergence message.

### `nfev`

Number of objective-function evaluations for the selected interval.

### `n_obs`

Number of paired pseudo-observations used in the fit.

### `interval`

Search interval that produced the selected optimum.

### `raw_result`

Original SciPy `OptimizeResult` associated with the selected interval.

This field is primarily for diagnostics.

### `interval_results`

Tuple containing the raw SciPy result for every searched interval.

## Example

```python
from pycopula_ml.copulas import FrankCopula
from pycopula_ml.estimation import fit_copula_mle
from pycopula_ml.marginals import bivariate_pseudo_observations


u, v = bivariate_pseudo_observations(x, y)

result = fit_copula_mle(
    FrankCopula,
    u,
    v,
    bounds=(
        (-50.0, -1e-6),
        (1e-6, 50.0),
    ),
)

print(f"theta_hat: {result.theta:.8f}")
print(f"log-likelihood: {result.log_likelihood:.8f}")
print(f"observations: {result.n_obs}")
print(f"selected interval: {result.interval}")
```
