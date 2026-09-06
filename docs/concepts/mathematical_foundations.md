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

It is not a different distribution. It is the same density represented on a logarithmic scale.

The package names this operation `logpdf()`.

## Likelihood

Given independent observations

$$
(u_1,v_1),\ldots,(u_n,v_n),
$$

the copula likelihood is

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

This form is preferred computationally because it converts products into sums and reduces numerical underflow.

## Future maximum-likelihood estimation

The future estimator will solve

$$
\widehat\theta
=
\arg\max_\theta
\ell(\theta).
$$

Equivalently, numerical optimizers can minimize

$$
-\ell(\theta).
$$

This is the next major statistical component planned for the package.
