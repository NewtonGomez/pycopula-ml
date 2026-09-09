# Parameter estimation

## Purpose

The `pycopula_ml.estimation` package estimates copula parameters after the
marginal variables have been transformed to pseudo-observations.

The current workflow is

$$
(X,Y)
\rightarrow
(U,V)
\rightarrow
\widehat{\theta}.
$$

The responsibilities are deliberately separated:

- `pycopula_ml.marginals` transforms original measurements into values on the
  unit interval;
- `pycopula_ml.copulas` evaluates a copula for a supplied parameter;
- `pycopula_ml.estimation` searches for the parameter that best explains the
  observed dependence under the selected copula family.

## Maximum-likelihood principle

Given paired pseudo-observations

$$
(u_1,v_1),\ldots,(u_n,v_n),
$$

and a copula density \(c_\theta\), the sample copula log-likelihood is

$$
\ell(\theta)
=
\sum_{i=1}^{n}
\log c_\theta(u_i,v_i).
$$

The estimator seeks

$$
\widehat{\theta}
=
\operatorname*{argmax}_{\theta}
\ell(\theta).
$$

The implementation uses a numerical minimizer, so it minimizes the equivalent
objective

$$
Q(\theta)
=
-\ell(\theta).
$$

Therefore,

$$
\operatorname*{argmin}_{\theta}
Q(\theta)
=
\operatorname*{argmax}_{\theta}
\ell(\theta).
$$

## Why the optimizer is outside the copula classes

A copula object answers questions such as:

```python
copula.cdf(u, v)
copula.pdf(u, v)
copula.logpdf(u, v)
copula.log_likelihood(u, v)
```

Those operations assume that `theta` is already known.

Parameter estimation is a different responsibility. It repeatedly creates
candidate copulas, evaluates their log-likelihoods, and compares the resulting
objective values.

Keeping these responsibilities separate avoids embedding optimization policy
inside each copula family.

## Pseudo-observations

The estimator expects paired one-dimensional arrays satisfying

$$
0<u_i<1,
\qquad
0<v_i<1.
$$

The open interval is intentional. Exact values of \(0\) or \(1\) can create
singular or numerically unstable log-density evaluations for several copula
families.

Original measurements should therefore first be transformed through either:

- fitted marginal CDFs; or
- rank-based pseudo-observations.

For example:

```python
from pycopula_ml.marginals import bivariate_pseudo_observations

u, v = bivariate_pseudo_observations(x, y)
```

## Likelihood vs pseudo-likelihood

When \(u_i\) and \(v_i\) are observed uniforms under known marginals,
maximizing the copula log-likelihood is ordinary maximum-likelihood estimation
for the copula parameter.

When \(u_i\) and \(v_i\) are estimated from the same sample, for example by
ranks,

$$
u_i=\frac{R_i}{n+1},
\qquad
v_i=\frac{S_i}{n+1},
$$

the procedure is commonly described more precisely as a copula
**pseudo-likelihood** or maximum pseudo-likelihood approach.

The current `fit_copula_mle()` API maximizes the supplied copula
log-likelihood in either case. It does not attempt to infer how the
pseudo-observations were constructed.

## Disjoint parameter domains

Some copula families have parameter spaces that are not represented by one
closed numerical interval.

For the Frank family,

$$
\theta\in\mathbb R\setminus\{0\},
$$

while independence is obtained only in the limit

$$
\theta\rightarrow0.
$$

Because `FrankCopula(theta=0)` is not part of the direct parameterization, the
estimator can search both sides of zero separately:

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

Each interval is optimized independently and the solution with the largest
log-likelihood is selected.

## Current scope

The estimation layer currently supports:

- one scalar dependence parameter;
- bounded one-dimensional numerical optimization;
- one interval or several disjoint intervals;
- structured optimization results;
- validation of pseudo-observations and bounds.

It does not yet provide:

- standard errors or confidence intervals;
- Hessian-based uncertainty estimates;
- multivariate parameter optimization;
- automatic copula-family selection;
- automatic parameter-domain discovery from a copula class;
- rank-inversion estimators such as inversion of Kendall's tau.
