# Marginal models

## Why marginals are required

Copulas operate on uniform variables, not directly on arbitrary measurements.

Given data \(X\),

$$
X
\rightarrow
F_X
\rightarrow
U=F_X(X).
$$

The `marginals` package provides two strategies:

1. parametric marginal estimation;
2. empirical/rank-based transformation.

## Public hierarchy

```text
MarginalModel
├── EmpiricalMarginal
└── ContinuousMarginalModel
    └── ParametricMarginal
```

## `MarginalModel`

Common operations:

- `fit(values)`
- `cdf(values)`
- `transform(values)`

The generic transform applies

$$
u_i=F_X(x_i)
$$

and clips exact boundary values to avoid numerical problems in later copula calculations.

## `ContinuousMarginalModel`

Adds:

- `pdf(values)`
- `logpdf(values)`

This interface is appropriate for continuous parametric distributions.

## Why empirical marginals are separate

An empirical CDF is a step function.

It does not have an ordinary continuous density equivalent to the PDFs provided by SciPy continuous distributions.

Therefore `EmpiricalMarginal` derives from `MarginalModel`, not `ContinuousMarginalModel`.

## Parametric path

$$
X
\rightarrow
\widehat F_X^{\text{parametric}}
\rightarrow
U.
$$

Example:

```python
from pycopula_ml.marginals import MarginalSelector

selector = MarginalSelector().fit(x)
u = selector.transform(x)
```

## Rank-based path

$$
X
\rightarrow
R_X
\rightarrow
U=\frac{R_X}{n+1}.
$$

Example:

```python
from pycopula_ml.marginals import pseudo_observations

u = pseudo_observations(x)
```

The rank-based route avoids assuming a theoretical marginal family.
