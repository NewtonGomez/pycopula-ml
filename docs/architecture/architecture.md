# Architecture

## Objective

The architecture separates four statistical responsibilities:

1. **marginal modeling**;
2. **dependence modeling with copulas**;
3. **parameter estimation**;
4. future **classification** layers.

The current package layout is:

```text
src/pycopula_ml/
├── copulas/
│   ├── base.py
│   └── frank.py
├── estimation/
│   └── maximum_likelihood.py
└── marginals/
    ├── base.py
    ├── empirical.py
    ├── parametric.py
    ├── result.py
    └── selection.py
```

## Statistical flow

For two original variables \(X\) and \(Y\),

$$
X
\rightarrow
\widehat F_X
\rightarrow
U,
$$

$$
Y
\rightarrow
\widehat F_Y
\rightarrow
V.
$$

The dependence model then evaluates

$$
(U,V,\theta)
\rightarrow
C_\theta(U,V).
$$

If \(\theta\) is unknown, the estimation layer uses the sample
log-likelihood

$$
\ell(\theta)
=
\sum_{i=1}^{n}
\log c_\theta(u_i,v_i)
$$

to obtain

$$
\widehat{\theta}
=
\operatorname*{argmax}_{\theta}
\ell(\theta).
$$

The full fitting workflow is therefore

```text
original data
    ↓
marginal transformation
    ↓
pseudo-observations (u, v)
    ↓
copula log-likelihood
    ↓
parameter estimation
    ↓
theta_hat
```

This separation is deliberate.

A copula should not need to know whether \(U\) and \(V\) came from:

- a Normal marginal;
- a Gamma marginal;
- a Lognormal marginal;
- an empirical CDF;
- rank-based pseudo-observations.

Likewise, the estimation layer should not contain the analytical density
formula for every copula family. It operates through the common copula
interface.

## Copula hierarchy

```text
BivariateCopula
└── FrankCopula
```

`BivariateCopula` defines common operations:

- `cdf()`
- `pdf()`
- `logpdf()`
- `log_likelihood()`
- common input preparation and validation

`FrankCopula` provides the Frank-specific formulas and parameter validation.

Future families can derive from the same interface.

## Marginal hierarchy

```text
MarginalModel
├── EmpiricalMarginal
└── ContinuousMarginalModel
    └── ParametricMarginal
```

`MarginalModel` defines:

- `fit()`
- `cdf()`
- `transform()`

`ContinuousMarginalModel` additionally requires:

- `pdf()`
- `logpdf()`

This distinction matters because an empirical CDF is a step function and does
not have an ordinary continuous density suitable for the same PDF interface.

## Model-selection layer

```text
MarginalSelector
├── MarginalCandidate
├── ParametricMarginal
└── MarginalFitResult
```

`MarginalSelector` is responsible for:

- fitting multiple candidate distributions;
- recording failed candidates;
- ranking successful candidates;
- selecting by AIC, BIC, or KS statistic;
- producing warnings;
- transforming observations with the selected marginal.

## Estimation layer

```text
fit_copula_mle()
└── CopulaFitResult
```

The estimation layer is responsible for:

- validating pseudo-observations for likelihood evaluation;
- evaluating candidate copula log-likelihoods;
- minimizing the negative log-likelihood;
- searching one or several bounded parameter intervals;
- selecting the best successful solution;
- returning statistical and numerical diagnostics.

It is intentionally generic with respect to the copula family.

The estimator requires a copula class or factory that can be called as

```python
copula(theta=<candidate>)
```

and whose instances implement

```python
log_likelihood(u, v)
```

This is why user code passes

```python
FrankCopula
```

rather than an already instantiated

```python
FrankCopula(theta=...)
```

to `fit_copula_mle()`.

## Why estimation is not under `tools`

A generic package path such as

```text
pycopula_ml.tools
```

does not communicate the statistical responsibility of the code.

Maximum-likelihood fitting belongs under

```text
pycopula_ml.estimation
```

because it is part of the public statistical model-fitting API, not merely an
internal utility.

This also leaves room for future estimators such as:

```text
estimation/
├── maximum_likelihood.py
├── rank_based.py
└── ...
```

without mixing statistical estimation with unrelated helpers.

## Design rule

The classifier planned for later should not contain logic such as:

```python
if family == "frank":
    ...
elif family == "clayton":
    ...
```

It should operate through a common copula interface.

Similarly:

- the copula layer should not decide which marginal distribution produced
  \(U\) and \(V\);
- the estimation layer should not reimplement family-specific copula
  mathematics;
- user scripts should not depend directly on the SciPy optimization backend.

## Separation of calculation and presentation

Core statistical classes and functions return values and structured results.

They should not depend on:

- `rich`;
- terminal formatting;
- plotting;
- interactive input.

Presentation utilities can be added separately later.

This keeps the statistical core usable from scripts, notebooks, tests,
services, and future classifier components.
