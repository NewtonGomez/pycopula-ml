# Gumbel-Hougaard copula

The bivariate Gumbel-Hougaard copula is

\[
C_\theta(u,v)
=
\exp\left[
-
\left(
(-\log u)^\theta
+
(-\log v)^\theta
\right)^{1/\theta}
\right],
\]

with

\[
\theta\ge1.
\]

Define

\[
x=-\log u,\qquad
y=-\log v,\qquad
s=x^\theta+y^\theta,\qquad
t=s^{1/\theta}.
\]

The density is

\[
c_\theta(u,v)
=
e^{-t}
(t+\theta-1)
s^{1/\theta-2}
(xy)^{\theta-1}
(uv)^{-1}.
\]

## Independence

\[
\theta=1
\quad\Longrightarrow\quad
C(u,v)=uv,\qquad c(u,v)=1.
\]

## Example

```python
from pycopula_lm.copulas import GumbelCopula

copula = GumbelCopula(theta=1.8)

cdf = copula.cdf(0.4, 0.7)
pdf = copula.pdf(0.4, 0.7)
```
