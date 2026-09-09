# Empirical marginals

## Empirical CDF

For observations

$$
X_1,\ldots,X_n,
$$

the empirical cumulative distribution function is

$$
\widehat F_n(x)
=
\frac{1}{n}
\sum_{i=1}^n
\mathbf 1(X_i\leq x).
$$

`EmpiricalMarginal` stores the sample and evaluates this step function efficiently using sorted observations and `numpy.searchsorted`.

## Example

```python
import numpy as np

from pycopula_ml.marginals import EmpiricalMarginal


x = np.array([10.0, 20.0, 30.0, 40.0])

marginal = EmpiricalMarginal().fit(x)

print(marginal.cdf([5.0, 10.0, 25.0, 40.0]))
```

Expected ECDF values are

$$
[0,\;0.25,\;0.5,\;1].
$$

## Pseudo-observations

For copula estimation, the package also provides rank-based pseudo-observations:

$$
u_i
=
\frac{R_i}{n+1}.
$$

The \(n+1\) denominator keeps the transformed observations away from the exact value \(1\).

Example:

```python
from pycopula_ml.marginals import pseudo_observations

u = pseudo_observations(x)
```

## Ties

The default ranking method is average rank.

For

$$
[10,20,20,40],
$$

the two tied values receive the same average rank.

Alternative ranking behavior can be requested through the `ties` argument where supported by `scipy.stats.rankdata`.

## Bivariate pseudo-observations

```python
from pycopula_ml.marginals import (
    bivariate_pseudo_observations,
)

u, v = bivariate_pseudo_observations(x, y)
```

Both samples must contain the same number of observations.

Each marginal is ranked independently.

## ECDF vs pseudo-observations

These are related but not identical.

For the largest fitted observation,

$$
\widehat F_n(x_{\max})=1.
$$

Rank-based pseudo-observations instead yield

$$
u_{\max}
=
\frac{n}{n+1}
<1.
$$

For copula pseudo-likelihood estimation, the rank transformation is usually the more direct tool.
