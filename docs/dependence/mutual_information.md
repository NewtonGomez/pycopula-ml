# Copula-based mutual information

## Purpose

`pycopula_ml.dependence.mutual_information()` evaluates the **mutual
information implied by an already parameterized, continuous bivariate
copula**. It does not estimate the copula parameter from data.

For an explanation of the underlying dependence measure, see
[Dependence measures](../concepts/dependence.md).

## Mathematical definition

For jointly continuous $X$ and $Y$,

$$
I(X;Y)
=\int\!\!\int f_{XY}(x,y)
\log\left(\frac{f_{XY}(x,y)}{f_X(x)f_Y(y)}\right)
\,dx\,dy.
$$

By Sklar's theorem, with $u=F_X(x)$ and $v=F_Y(y)$,

$$
f_{XY}(x,y)
=c\bigl(F_X(x),F_Y(y)\bigr)f_X(x)f_Y(y).
$$

Under the usual regularity conditions, this gives

$$
\boxed{
I(X;Y)
=\int_0^1\!\int_0^1
c(u,v)\log c(u,v)\,du\,dv
}.
$$

For a fitted parametric copula $c_{\widehat\theta}$, the value returned by
the implementation is the **model-based** quantity

$$
\widehat I_C
=\int_0^1\!\int_0^1
c_{\widehat\theta}(u,v)
\log c_{\widehat\theta}(u,v)\,du\,dv.
$$

It is not necessarily equal to the unknown population MI if the copula family
is misspecified or the fitted parameter is imprecise.

!!! note "Interpretation"
    A positive MI indicates dependence in the fitted model. MI cannot indicate
    whether the association is positive or negative, and a value of `0.4`
    nats does not mean 40% correlation. The measure may exceed one nat.

## Workflow from original observations

Given paired measurements $x_1,\ldots,x_n$ and $y_1,\ldots,y_n$:

1. Obtain pseudo-observations $u_i,v_i\in(0,1)$ from fitted marginal CDFs or
   rank-based transformations.
2. Fit a copula parameter $\widehat\theta$ by maximizing the sample
   log-likelihood.
3. Instantiate the copula with the estimated parameter.
4. Evaluate the mutual-information integral for that fitted copula.

For rank-based pseudo-observations, a common choice is

$$
u_i=\frac{\operatorname{rank}(x_i)}{n+1},
\qquad
v_i=\frac{\operatorname{rank}(y_i)}{n+1}.
$$

The denominator `n + 1` keeps transformed ranks strictly inside the
open unit interval. For tied values, an explicit ranking method ensures
reproducible pseudo-observations.

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas.parametrics.frank import FrankCopula
from pycopula_ml.dependence import mutual_information
from pycopula_ml.estimation import fit_copula_mle

x = np.array([10, 15, 20, 25, 30], dtype=float)
y = np.array([20, 35, 30, 45, 50], dtype=float)

u = rankdata(x, method="average") / (len(x) + 1)
v = rankdata(y, method="average") / (len(y) + 1)

fit = fit_copula_mle(
    FrankCopula,
    u,
    v,
    bounds=((-20.0, -1e-4), (1e-4, 20.0)),
)

fitted_copula = FrankCopula(theta=fit.theta)
mi = mutual_information(fitted_copula, order=64)

print(f"theta: {fit.theta:.6f}")
print(f"log-likelihood: {fit.log_likelihood:.6f}")
print(f"mutual information: {mi:.6f} nats")
```

!!! warning "Illustrative dataset"
    Five observations are sufficient to demonstrate the API but are generally
    insufficient for a reliable estimate of dependence. Samples must contain
    paired observations. Many ties or constant features require additional
    statistical handling.

`fit_copula_mle()` operates on pseudo-observations and is documented
separately in [Maximum-likelihood estimation](../estimation/maximum_likelihood.md).
The two disjoint intervals shown above avoid the Frank parameter's
independence boundary; valid parameter domains depend on the family and its
implementation.

## Numerical integration

For a general fitted continuous copula without a dedicated analytical
formula, the implementation performs tensor-product Gauss--Legendre
quadrature, using `copula.logpdf(u, v)` rather than taking the logarithm of
an independently calculated PDF:

$$
g(u,v)
=\exp\bigl(\log c(u,v)\bigr)\log c(u,v).
$$

With $q$ quadrature nodes and weights on $(0,1)$,

$$
\widehat I_C^{(q)}
=\sum_{i=1}^{q}\sum_{j=1}^{q}
w_iw_jc(u_i,u_j)\log c(u_i,u_j).
$$

The integral requires $q^2$ log-density evaluations. The quadrature order is
controlled by `order` (default `64`, minimum `8` for the numerical branch).

- The integrand is treated as zero when the density vanishes, using the limit
  $c\log c\to0$ as $c\to0^+$.
- A nonfinite integrand raises `FloatingPointError`.
- A materially negative numerical integral raises `ValueError`; tiny negative
  rounding error is clamped to zero.
- Larger `order` may improve numerical accuracy but increases computation.

!!! warning "Convergence and tail dependence"
    A fixed integration order does not guarantee convergence, especially for
    copulas with strong tail dependence or near-boundary parameters. Compare
    `order=64`, `128`, and `256` on representative fitted copulas and examine
    stability. Quadrature integrates the model, not the empirical sample.

## Gaussian copula: exact result

For a bivariate Gaussian copula with correlation parameter $\rho$,

$$
\boxed{
I(X;Y)=-\frac{1}{2}\log(1-\rho^2)
},\qquad -1<\rho<1.
$$

The Gaussian implementation avoids quadrature:

```python
from pycopula_ml.dependence import gaussian_mutual_information

mi = gaussian_mutual_information(rho=0.6)
print(mi)  # approximately 0.2231435513 nats
```

Passing a `GaussianCopula` instance to `mutual_information()` uses the same
formula, treating `copula.theta` as the Gaussian correlation parameter.
Consequently, Gaussian correlations of $0.6$ and $-0.6$ have identical MI.
The `order` argument does not control the Gaussian closed-form calculation.

## Mutual information versus log-likelihood

| Quantity | Definition | Role |
| --- | --- | --- |
| Fitted parameter | $\widehat\theta=\arg\max_\theta\ell(\theta)$ | Describes dependence in a specified copula family |
| Sample log-likelihood | $\ell(\widehat\theta)=\sum_k\log c_{\widehat\theta}(u_k,v_k)$ | Evaluates the fitted model on the observed pairs |
| Model-based mutual information | $\int_0^1\!\int_0^1c_{\widehat\theta}\log c_{\widehat\theta}$ | Measures dependence implied by the fitted copula |

The log-likelihood is a **sum over observations**. MI is an
**integral over the unit square**. They are not interchangeable and need not
have the same scale.

## Next steps

- [Pairwise dependence matrices](weight_matrices.md): repeat this computation
  across feature pairs and classes.
- [Dependence API](../api/dependence.md): function signatures, results, and
  validations.
- [Dependence validation](../testing/dependence_validation.md): numerical and
  model-based tests.
- [Examples](../examples/dependence.md): complete Python examples.
