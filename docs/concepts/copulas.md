# Copulas

## Purpose

The `pycopula_ml.copulas` package models dependence after marginal variables
have been transformed to the unit interval.

For CDF and density evaluation, copula inputs belong to

$$
u,v\in[0,1].
$$

For parameter estimation through log-likelihood, the current estimation layer
requires pseudo-observations strictly inside

$$
0<u,v<1.
$$

Original measurements \(X,Y\) should not be passed directly to a copula unless
they are already uniform.

## Public hierarchy

```text
BivariateCopula
└── FrankCopula
```

## `BivariateCopula`

The abstract base class defines a common interface for continuous bivariate
copulas.

Required subclass implementations:

```python
cdf(u, v)
pdf(u, v)
```

Default/common functionality:

```python
logpdf(u, v)
log_likelihood(u, v)
```

The default `logpdf()` can be computed as

```python
np.log(self.pdf(u, v))
```

but subclasses may override it with a direct analytical expression for
numerical stability.

## Common input preparation

`BivariateCopula._prepare_inputs()` performs common validation before
family-specific mathematics is evaluated.

It:

1. converts inputs to NumPy floating-point arrays;
2. broadcasts compatible shapes;
3. rejects non-finite values;
4. verifies that all values belong to \([0,1]\).

This logic belongs in the base class because it applies to all copula families.

Family-specific parameter restrictions remain in each subclass.

## Sample log-likelihood

For a copula parameterized by \(\theta\), the generic sample log-likelihood is

$$
\ell(\theta)
=
\sum_i \log c_\theta(u_i,v_i).
$$

The density itself changes by family; the likelihood construction does not.

This common method is the interface used by the estimation layer:

```text
fit_copula_mle()
        ↓
copula(theta=candidate)
        ↓
log_likelihood(u, v)
```

The copula class therefore contains the probability model, while
`pycopula_ml.estimation` contains the numerical fitting strategy.

## Current family support

Implemented:

- [Frank](../copulas/frank_copula.md)

Planned:

- Clayton
- Gumbel
- Joe
- AMH
- Gaussian
- Independence
- later Student-t and additional families

## Parameter estimation

Copula objects are instantiated with known parameters:

```python
copula = FrankCopula(theta=2.0)
```

When the parameter is unknown, use the estimation package instead:

```python
from pycopula_ml.estimation import fit_copula_mle

result = fit_copula_mle(
    FrankCopula,
    u,
    v,
    bounds=(
        (-50.0, -1e-6),
        (1e-6, 50.0),
    ),
)

theta_hat = result.theta
```

This avoids placing optimizer-specific logic inside `FrankCopula`.
