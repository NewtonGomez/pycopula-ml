# Ali-Mikhail-Haq copula

The Ali-Mikhail-Haq (AMH) copula is

\[
C_\theta(u,v)
=
\frac{uv}{
1-\theta(1-u)(1-v)
},
\]

with

\[
-1\le\theta<1.
\]

Let

\[
d=1-\theta(1-u)(1-v).
\]

The density is

\[
c_\theta(u,v)
=
\frac{
1+
\theta\left[(1+u)(1+v)-3\right]
+
\theta^2(1-u)(1-v)
}{
d^3
}.
\]

## Independence

\[
\theta=0
\quad\Longrightarrow\quad
C(u,v)=uv,\qquad c(u,v)=1.
\]

## Example

```python
from pycopula_lm.copulas import AliMikhailHaqCopula

copula = AliMikhailHaqCopula(theta=0.5)

cdf = copula.cdf(0.4, 0.7)
pdf = copula.pdf(0.4, 0.7)
```
