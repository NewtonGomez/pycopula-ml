# Clayton copula

The bivariate Clayton copula is

\[
C_\theta(u,v)
=
\left[
\max\left(
u^{-\theta}+v^{-\theta}-1,\,
0
\right)
\right]^{-1/\theta}.
\]

For the continuous-density implementation in `pycopula_lm`,

\[
\theta>-1,
\]

with \(\theta=0\) interpreted as independence.

The density, where
\(s=u^{-\theta}+v^{-\theta}-1>0\), is

\[
c_\theta(u,v)
=
(1+\theta)
(uv)^{-(1+\theta)}
s^{-2-1/\theta}.
\]

When \(s\le0\), the density is zero.

## Independence

\[
\theta=0
\quad\Longrightarrow\quad
C(u,v)=uv,\qquad c(u,v)=1.
\]

The point \(\theta=-1\) is not accepted because it gives the singular lower
Fréchet bound rather than an ordinary absolutely continuous density.

## Example

```python
from pycopula_lm.copulas import ClaytonCopula

copula = ClaytonCopula(theta=2.0)

cdf = copula.cdf(0.4, 0.7)
pdf = copula.pdf(0.4, 0.7)
log_pdf = copula.logpdf(0.4, 0.7)
```
