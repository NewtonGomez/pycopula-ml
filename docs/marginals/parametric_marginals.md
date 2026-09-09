# Parametric marginals

## Purpose

`ParametricMarginal` wraps continuous distributions from `scipy.stats`.

The package does **not** reimplement Normal, Gamma, Lognormal, Weibull, or Exponential distributions.

SciPy remains responsible for:

- parameter fitting;
- PDF evaluation;
- log-PDF evaluation;
- CDF evaluation;
- distribution support.

`pycopula-ml` adds a consistent interface and diagnostics needed by the copula workflow.

## Example: Normal

```python
from scipy.stats import norm

from pycopula_ml.marginals import ParametricMarginal


marginal = ParametricMarginal(
    norm,
    name="normal",
)

marginal.fit(x)

u = marginal.transform(x)
```

For a Normal model, SciPy uses

$$
\mu=\text{loc},
\qquad
\sigma=\text{scale}.
$$

## Example: Gamma with fixed location

```python
from scipy.stats import gamma

from pycopula_ml.marginals import ParametricMarginal


marginal = ParametricMarginal(
    gamma,
    name="gamma",
    fit_kwargs={
        "floc": 0.0,
    },
)

marginal.fit(x)
```

Fixing `loc=0` is a modeling assumption and should only be used when the variable's support justifies it.

## Fit result

A successful fit stores a `MarginalFitResult` containing:

- distribution name;
- fitted parameters;
- sample size;
- number of free parameters;
- log-likelihood;
- AIC;
- BIC;
- KS statistic;
- exploratory KS p-value.

## Information criteria

For \(k\) free parameters,

$$
AIC
=
2k-2\ell,
$$

and

$$
BIC
=
k\log n-2\ell.
$$

Lower values are preferred within the candidate set.

Fixed parameters are excluded from the free-parameter count when possible.

## Probability integral transform

After fitting,

```python
u = marginal.transform(x)
```

computes

$$
u_i=\widehat F_X(x_i)
$$

and clips numerical boundaries into the open unit interval.
