# Frank copula

## Definition

The bivariate Frank copula is

$$
C_\theta(u,v)
=
-\frac{1}{\theta}
\ln
\left(
1+
\frac{
(e^{-\theta u}-1)(e^{-\theta v}-1)
}{
e^{-\theta}-1
}
\right),
$$

for

$$
\theta\in\mathbb R\setminus\{0\}.
$$

Positive \(\theta\) represents positive dependence and negative \(\theta\)
represents negative dependence.

Independence is obtained as the limiting case

$$
\theta\to0.
$$

## Density

The copula density is

$$
c_\theta(u,v)
=
\frac{
\theta(1-e^{-\theta})e^{-\theta(u+v)}
}{
\left[
(1-e^{-\theta})
-
(1-e^{-\theta u})(1-e^{-\theta v})
\right]^2
}.
$$

It is obtained from

$$
c_\theta(u,v)
=
\frac{\partial^2 C_\theta(u,v)}
{\partial u\,\partial v}.
$$

## Log-density

The direct log-density can be written as

$$
\log c_\theta(u,v)
=
\log\left[
\theta(1-e^{-\theta})
\right]
-\theta(u+v)
-
2\log
\left|
(1-e^{-\theta})
-
(1-e^{-\theta u})(1-e^{-\theta v})
\right|.
$$

Direct log-density evaluation is useful because parameter estimation works with
sums of log-densities.

## Sample log-likelihood

For paired pseudo-observations

$$
(u_1,v_1),\ldots,(u_n,v_n),
$$

the Frank sample log-likelihood is

$$
\ell(\theta)
=
\sum_{i=1}^{n}
\log c_\theta(u_i,v_i).
$$

The class exposes this through:

```python
copula.log_likelihood(u, v)
```

when `copula` is a `FrankCopula` instantiated at a specific candidate
\(\theta\).

## Numerical implementation

The implementation uses NumPy operations such as:

- `np.expm1(x)` for \(e^x-1\);
- `np.log1p(x)` for \(\log(1+x)\).

These forms improve numerical accuracy near zero.

## Parameter validation

`theta=0` is rejected by the direct implementation because the formula contains
\(1/\theta\).

The mathematical limit at zero corresponds to the independence copula but is
not currently represented by `FrankCopula(theta=0)`.

## Parameter estimation

When \(\theta\) is unknown, use the estimation layer.

Because Frank excludes exactly zero, search the negative and positive parameter
regions separately:

```python
from pycopula_ml.copulas import FrankCopula
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

Internally, each candidate value is evaluated through

$$
-\ell(\theta),
$$

and the best successful interval is selected.

The final result reports the statistical log-likelihood directly:

```python
print(result.theta)
print(result.log_likelihood)
print(result.interval)
```

For Frank:

- `result.theta > 0` indicates positive dependence;
- `result.theta < 0` indicates negative dependence;
- an estimate numerically close to zero indicates approximate independence.

## Complete workflow from original data

For raw paired measurements:

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

fitted_copula = FrankCopula(theta=result.theta)
```

This separates:

```text
x, y
  ↓
marginal/rank transformation
  ↓
u, v
  ↓
MLE
  ↓
theta_hat
  ↓
fitted FrankCopula
```

## Direct evaluation example

If \(\theta\) is already known:

```python
from pycopula_ml.copulas import FrankCopula


copula = FrankCopula(theta=2.0)

cdf = copula.cdf(0.3, 0.7)
pdf = copula.pdf(0.3, 0.7)
logpdf = copula.logpdf(0.3, 0.7)
```

## Mathematical tests

The test suite checks properties including:

$$
C(u,0)=0,
$$

$$
C(0,v)=0,
$$

$$
C(u,1)=u,
$$

$$
C(1,v)=v,
$$

symmetry,

$$
C(u,v)=C(v,u),
$$

positive density,

$$
c(u,v)>0,
$$

and consistency,

$$
\operatorname{logpdf}(u,v)
=
\log(\operatorname{pdf}(u,v)).
$$

The estimation layer is tested separately so failures in numerical fitting can
be distinguished from failures in the Frank probability formulas.
