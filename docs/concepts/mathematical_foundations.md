# Mathematical foundations

## Marginal distributions

For a random variable \(X\), its cumulative distribution function is

$$
F_X(x)=P(X\leq x).
$$

For a continuous distribution, its density is

$$
f_X(x)=\frac{dF_X(x)}{dx}.
$$

A probability over an interval is obtained by integration:

$$
P(a<X<b)
=
\int_a^b f_X(x)\,dx.
$$

## Bivariate distribution

For two random variables \(X\) and \(Y\),

$$
F_{X,Y}(x,y)
=
P(X\leq x,\;Y\leq y).
$$

If the joint distribution is continuous,

$$
f_{X,Y}(x,y)
=
\frac{\partial^2F_{X,Y}(x,y)}
{\partial x\,\partial y}.
$$

## Probability integral transform

For a continuous variable \(X\),

$$
U=F_X(X)
$$

is Uniform on \((0,1)\) under the ideal population model.

Similarly,

$$
V=F_Y(Y).
$$

This transformation is the bridge between real-world variables and copulas.

## Sklar's theorem

For continuous marginal distributions,

$$
F_{X,Y}(x,y)
=
C(F_X(x),F_Y(y)),
$$

or

$$
F_{X,Y}(x,y)
=
C(u,v).
$$

At the density level,

$$
f_{X,Y}(x,y)
=
c(u,v)
f_X(x)
f_Y(y).
$$

This separates:

- marginal behavior \(f_X,f_Y\);
- dependence behavior \(c\).

## Copula CDF

A bivariate copula is a CDF on the unit square:

$$
C(u,v)
=
P(U\leq u,\;V\leq v).
$$

It satisfies properties including

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
C(1,v)=v.
$$

## Copula density

For a continuous bivariate copula,

$$
c(u,v)
=
\frac{\partial^2C(u,v)}
{\partial u\,\partial v}.
$$

This is what the package names `pdf()`.

## Log-density

The log-density is

$$
\log c(u,v).
$$

It is not a different distribution. It is the same density represented on a
logarithmic scale.

The package names this operation `logpdf()`.

## Likelihood

Given paired observations

$$
(u_1,v_1),\ldots,(u_n,v_n),
$$

the copula likelihood for a parameter \(\theta\) is

$$
L(\theta)
=
\prod_{i=1}^n
c_\theta(u_i,v_i).
$$

The log-likelihood is

$$
\ell(\theta)
=
\sum_{i=1}^n
\log c_\theta(u_i,v_i).
$$

This form is preferred computationally because it converts products into sums
and reduces numerical underflow.

In the package:

```python
copula.log_likelihood(u, v)
```

returns this summed log-density for the current copula parameter.

## Maximum-likelihood estimation

The current scalar estimator solves

$$
\widehat\theta
=
\operatorname*{argmax}_\theta
\ell(\theta).
$$

The implementation uses a numerical minimizer on the equivalent objective

$$
Q(\theta)
=
-\ell(\theta).
$$

Therefore,

$$
\widehat\theta
=
\operatorname*{argmin}_\theta
[-\ell(\theta)].
$$

In code:

```python
from pycopula_ml.estimation import fit_copula_mle

result = fit_copula_mle(
    CopulaClass,
    u,
    v,
    bounds=(lower, upper),
)

theta_hat = result.theta
```

## Why pseudo-observations must stay inside the open unit interval

A copula CDF is defined on

$$
[0,1]^2.
$$

However, likelihood estimation uses the copula density and log-density.

For numerical fitting, the estimator requires

$$
0<u_i<1,
\qquad
0<v_i<1.
$$

Rank-based pseudo-observations use

$$
u_i=\frac{R_i}{n+1},
$$

which naturally avoids the exact upper boundary \(1\).

## Copula likelihood and pseudo-likelihood

If the marginal distributions are known and \(U,V\) are exact transformed
uniform variables, maximizing

$$
\sum_i\log c_\theta(u_i,v_i)
$$

is ordinary copula maximum likelihood.

If the transformations are estimated from the same sample, especially through
ranks, the same objective is commonly called a copula pseudo-likelihood.

The current estimator maximizes the supplied copula log-likelihood without
trying to determine which marginal-estimation strategy generated \(u\) and
\(v\).

## Disjoint parameter spaces

A numerical optimizer normally searches a continuous bounded interval.

Some copula families require more than one search interval.

For Frank,

$$
\theta\in\mathbb R\setminus\{0\}.
$$

The two regions

$$
\theta<0
$$

and

$$
\theta>0
$$

can therefore be searched independently.

The estimator compares the optimized log-likelihoods from both regions and
selects the better result.
