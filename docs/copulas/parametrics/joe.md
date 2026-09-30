# Joe copula

The bivariate Joe copula is

\[
C_\theta(u,v)
=
1-
\left[
(1-u)^\theta
+
(1-v)^\theta
-
(1-u)^\theta(1-v)^\theta
\right]^{1/\theta},
\]

with

\[
\theta\ge1.
\]

Define

\[
a=(1-u)^\theta,\qquad
b=(1-v)^\theta,\qquad
s=a+b-ab.
\]

Its density is

\[
c_\theta(u,v)
=
(1-u)^{\theta-1}
(1-v)^{\theta-1}
s^{1/\theta-2}
\left[
\theta s+
(\theta-1)(1-a)(1-b)
\right].
\]

## Independence

\[
\theta=1
\quad\Longrightarrow\quad
C(u,v)=uv,\qquad c(u,v)=1.
\]

## Example

```python
from pycopula_lm.copulas import JoeCopula

copula = JoeCopula(theta=2.0)

cdf = copula.cdf(0.4, 0.7)
pdf = copula.pdf(0.4, 0.7)
```
