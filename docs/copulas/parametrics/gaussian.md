# Gaussian copula

The bivariate Gaussian copula is constructed from a standard bivariate normal
distribution with correlation parameter \(\rho\):

\[
C_\rho(u,v)
=
\Phi_\rho\left(
\Phi^{-1}(u),
\Phi^{-1}(v)
\right).
\]

In the common parametric API, the library stores this parameter as
`theta`, so

\[
\theta=\rho,\qquad -1<\theta<1.
\]

Let

\[
z_u=\Phi^{-1}(u),\qquad z_v=\Phi^{-1}(v).
\]

The density is

\[
c_\rho(u,v)
=
\frac{1}{\sqrt{1-\rho^2}}
\exp\left[
\frac{
2\rho z_u z_v
-\rho^2(z_u^2+z_v^2)
}{
2(1-\rho^2)
}
\right].
\]

## Independence

\[
\rho=0
\quad\Longrightarrow\quad
C(u,v)=uv,\qquad c(u,v)=1.
\]

## Numerical implementation

The density and log-density use the analytical formula. The CDF is evaluated
using SciPy's bivariate normal CDF.

## Example

```python
from pycopula_lm.copulas import GaussianCopula

copula = GaussianCopula(theta=0.6)

print(copula.rho)
print(copula.pdf(0.4, 0.7))
```
