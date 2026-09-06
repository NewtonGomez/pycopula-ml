# Copulas

## Purpose

The `pycopula_ml.copulas` package models dependence after marginal variables have been transformed to the unit interval.

Inputs are therefore expected to satisfy
$$
u,v\in[0,1].
$$
Original measurements \(X,Y\) should not be passed directly to a copula unless they are already uniform.

## Public hierarchy

```text
BivariateCopula
└── FrankCopula
```

## `BivariateCopula`

The abstract base class defines a common interface for continuous bivariate copulas.

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

but subclasses may override it with a direct analytical expression for numerical stability.

## Common input preparation

`BivariateCopula._prepare_inputs()` performs common validation before family-specific mathematics is evaluated.

It:

1. converts inputs to NumPy floating-point arrays;
2. broadcasts compatible shapes;
3. rejects non-finite values;
4. verifies that all values belong to \([0,1]\).

This logic belongs in the base class because it applies to all copula families.

Family-specific parameter restrictions remain in each subclass.

## Sample log-likelihood

The generic sample log-likelihood is

$$
\ell
=
\sum_i \log c(u_i,v_i).
$$

This operation is general across continuous copula families.

The density itself changes by family; the likelihood construction does not.

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
