# Copula API

## Public imports

```python
from pycopula_ml.copulas import (
    BivariateCopula,
    FrankCopula,
)
```

## `BivariateCopula`

Abstract base class for continuous bivariate copulas.

### `cdf(u, v)`

Evaluate

$$
C(u,v).
$$

Subclasses must implement this method.

### `pdf(u, v)`

Evaluate

$$
c(u,v)
=
\frac{\partial^2C(u,v)}
{\partial u\,\partial v}.
$$

Subclasses must implement this method.

### `logpdf(u, v)`

Evaluate

$$
\log c(u,v).
$$

The base implementation may use `np.log(pdf())`.

A subclass can override it with a numerically stable analytical expression.

### `log_likelihood(u, v)`

Evaluate

$$
\ell
=
\sum_i\log c(u_i,v_i).
$$

### `_prepare_inputs(u, v)`

Internal helper.

Responsibilities:

- convert to NumPy arrays;
- enforce floating-point representation;
- broadcast compatible inputs;
- reject non-finite values;
- enforce \([0,1]\) domain.

## `FrankCopula`

### Constructor

```python
FrankCopula(theta=2.0)
```

### Parameter

```text
theta : float
```

Requirements:

- finite;
- non-zero.

### Methods

```python
cdf(u, v)
pdf(u, v)
logpdf(u, v)
log_likelihood(u, v)
```

## Example

```python
import numpy as np

from pycopula_ml.copulas import FrankCopula


u = np.array([0.2, 0.4, 0.7])
v = np.array([0.3, 0.6, 0.8])

copula = FrankCopula(theta=2.0)

print(copula.cdf(u, v))
print(copula.pdf(u, v))
print(copula.logpdf(u, v))
print(copula.log_likelihood(u, v))
```
