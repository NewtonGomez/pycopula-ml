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

Positive \(\theta\) represents positive dependence and negative \(\theta\) represents negative dependence.

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

Direct log-density evaluation is useful because MLE works with sums of log-densities.

## Numerical implementation

The implementation uses NumPy operations such as:

- `np.expm1(x)` for \(e^x-1\);
- `np.log1p(x)` for \(\log(1+x)\).

These forms improve accuracy near zero.

## Parameter validation

`theta=0` is rejected by the direct implementation because the formula contains \(1/\theta\).

The mathematical limit at zero corresponds to the independence copula but is not yet implemented as a special branch.

## Example

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
