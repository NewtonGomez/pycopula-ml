# Dependence API

## Public imports

The installed package exposes the following functions and result type:

```python
from pycopula_ml.dependence import (
    CopulaWeightMatrices,
    build_weight_matrices,
    gaussian_mutual_information,
    mutual_information,
)
```


## `gaussian_mutual_information(rho)`

```python
gaussian_mutual_information(rho: float) -> float
```

Return the exact mutual information (in nats) for a bivariate Gaussian copula:

$$
I=-\tfrac{1}{2}\log(1-\rho^2),\qquad -1<\rho<1.
$$

**Parameters**

- `rho`: finite Gaussian copula correlation, strictly between `-1` and `1`.

**Returns**: Python `float`, in nats.

**Raises**: `ValueError` for a nonfinite or out-of-domain `rho`.

## `mutual_information(copula, *, order=64)`

```python
mutual_information(copula: object, *, order: int = 64) -> float
```

Evaluate the copula-based mutual information of an already instantiated
continuous bivariate copula. Gaussian copulas use the closed-form expression;
other copulas use tensor-product Gauss--Legendre quadrature over $(0,1)^2$.

**Parameters**

- `copula`: fitted bivariate copula object exposing `logpdf(u, v)`.
  Instances of `GaussianCopula` also provide `theta`, interpreted as `rho`.
- `order`: integer quadrature order per dimension (minimum `8` in the
  numerical branch, default `64`). Ignored for Gaussian copula evaluation.

**Returns**: Python `float`, in nats.

**Raises**:

- `ValueError`: unsupported numerical quadrature order, incompatible
  log-density output shape, or materially negative numerical integral.
- `FloatingPointError`: nonfinite or overflowing values in the quadrature
  computation.

```python
from pycopula_ml.copulas.parametrics.frank import FrankCopula
from pycopula_ml.dependence import mutual_information

copula = FrankCopula(theta=3.0)
mi = mutual_information(copula, order=64)
```

## `build_weight_matrices(...)`

```python
build_weight_matrices(
    U,
    copula_class,
    *,
    bounds,
    weight_method="mutual_information",
    feature_names=None,
    quadrature_order=64,
) -> CopulaWeightMatrices
```

Fit the chosen parametric bivariate copula to each pair of columns in the
**already transformed** matrix `U`. Return a symmetric matrix for each of
copula parameters, maximized log-likelihood, and mutual information.

**Parameters**

- `U`: 2D array of shape `(n_observations, n_features)` with at least two
  observations and two features; finite values in the open interval `(0,1)`.
- `copula_class`: class or factory accepting `theta=<candidate>` and providing
  `log_likelihood(u,v)` and `logpdf(u,v)`.
- `bounds`: required bounded search interval or sequence of disjoint
  intervals accepted by `fit_copula_mle()`.
- `weight_method`: either `"mutual_information"` (default) or `"likelihood"`.
- `feature_names`: optional sequence of unique feature names; inferred
  from `U.columns` for DataFrames, otherwise generated as `X1`, `X2`, etc.
- `quadrature_order`: integration order forwarded to `mutual_information()`.

**Returns**: `CopulaWeightMatrices`.

**Raises**: `ValueError` for invalid data or configuration;
`RuntimeError` if pairwise MLE fails; other numerical exceptions may
propagate from fitting or integration.

## `CopulaWeightMatrices`

Frozen dataclass with attributes:

```text
theta               : numpy.ndarray, shape (d, d)
log_likelihood      : numpy.ndarray, shape (d, d)
mutual_information  : numpy.ndarray, shape (d, d)
weights             : numpy.ndarray, shape (d, d)
feature_names       : tuple[str, ...]
weight_method       : str
```

All numeric matrices are symmetric and have a zero diagonal **by
construction**. The diagonal is a placeholder, not the MI of a variable with
itself. Although the dataclass is frozen, contained NumPy arrays may still
be modified by callers.

## Example

```python
from scipy.stats import rankdata
from pycopula_ml.copulas.parametrics.gaussian import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices

# x: array of shape (n_observations, n_features)
u = rankdata(x, axis=0, method="average") / (len(x) + 1)

result = build_weight_matrices(
    u,
    GaussianCopula,
    bounds=(-0.95, 0.95),
)

print(result.mutual_information)
```

See [Copula-based mutual information](../dependence/mutual_information.md)
and [Pairwise copula weight matrices](../dependence/weight_matrices.md) for
additional theory and statistical interpretation.
