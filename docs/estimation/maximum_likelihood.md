# Maximum-likelihood estimation

## Public interface

```python
from pycopula_ml.estimation import (
    CopulaFitResult,
    fit_copula_mle,
)
```

`fit_copula_mle()` estimates a scalar copula dependence parameter by maximizing
the copula log-likelihood.

## Basic workflow

For original paired observations \(x\) and \(y\):

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

print(result.theta)
print(result.log_likelihood)
```

The workflow is

$$
(x_i,y_i)
\rightarrow
(u_i,v_i)
\rightarrow
\ell(\theta)
\rightarrow
\widehat{\theta}.
$$

## Objective function

For every candidate parameter value, the estimator constructs

```python
copula(theta=theta)
```

and evaluates

```python
copula.log_likelihood(u, v)
```

which corresponds to

$$
\ell(\theta)
=
\sum_{i=1}^{n}
\log c_\theta(u_i,v_i).
$$

SciPy's scalar optimizer minimizes

$$
-\ell(\theta),
$$

and the public result converts the selected objective value back to the
maximized log-likelihood.

## One search interval

For a family whose admissible parameter region can be represented by one
bounded interval:

```python
result = fit_copula_mle(
    SomeCopula,
    u,
    v,
    bounds=(0.01, 20.0),
)
```

The lower bound must be strictly smaller than the upper bound.

## Multiple search intervals

Disjoint domains can be represented by a sequence of intervals:

```python
result = fit_copula_mle(
    FrankCopula,
    u,
    v,
    bounds=(
        (-50.0, -1e-6),
        (1e-6, 50.0),
    ),
)
```

The estimator:

1. optimizes the negative log-likelihood on each interval;
2. discards unsuccessful or non-finite solutions;
3. compares the successful objective values;
4. returns the solution with the largest log-likelihood.

This removes Frank-specific branching from user scripts.

## Result object

The return value is a `CopulaFitResult`.

Important fields are:

```python
result.theta
result.log_likelihood
result.success
result.message
result.nfev
result.n_obs
result.interval
result.raw_result
result.interval_results
```

### `theta`

Maximum-likelihood estimate:

$$
\widehat{\theta}.
$$

### `log_likelihood`

Maximized value

$$
\ell(\widehat{\theta}).
$$

This is returned with the statistical sign convention, even though the
underlying optimizer minimizes the negative log-likelihood.

### `interval`

The parameter interval that produced the selected optimum.

For a Frank search over both signs, this makes it explicit whether the selected
solution came from the negative or positive dependence region.

### `interval_results`

Raw SciPy results for every interval searched.

This is mainly useful for diagnostics and development. Most user code should
prefer the structured top-level fields.

## Input validation

The estimator requires `u` and `v` to be:

- one-dimensional;
- non-empty;
- equal in shape;
- finite;
- strictly inside \((0,1)\).

For example, exact boundary values are rejected:

```text
u = 0
u = 1
v = 0
v = 1
```

This is stricter than the generic copula CDF domain \([0,1]\) because
log-likelihood evaluation is an interior-density calculation.

## Optimization controls

The current function exposes:

```python
xatol
maxiter
```

`xatol` controls the absolute tolerance of the estimated scalar parameter.

`maxiter` controls the maximum number of optimizer iterations per search
interval.

The implementation currently uses SciPy's bounded scalar optimizer internally.
This is intentionally hidden behind the `pycopula_ml.estimation` API.

## Frank example and association direction

For the bivariate Frank copula:

$$
\widehat{\theta}>0
$$

indicates positive dependence, while

$$
\widehat{\theta}<0
$$

indicates negative dependence.

A fitted value very close to zero indicates approximate independence, even
though the direct Frank implementation excludes exactly

$$
\theta=0.
$$

Example:

```python
import numpy as np

theta_hat = result.theta

if np.isclose(theta_hat, 0.0, atol=1e-4):
    association = "approximately independent"
elif theta_hat > 0.0:
    association = "positive"
else:
    association = "negative"
```

## Numerical failures

If a candidate parameter produces a non-finite log-likelihood, the internal
objective treats that candidate as invalid.

If every searched interval fails to produce a finite successful optimization,
`fit_copula_mle()` raises `RuntimeError`.

This is preferable to silently returning a meaningless parameter estimate.

## Design note

The user-facing code does not import `scipy.optimize`.

That separation is intentional:

```text
user code
    ↓
pycopula_ml.estimation.fit_copula_mle
    ↓
SciPy optimization backend
```

The optimization backend can therefore change later without changing the
public estimation workflow.
