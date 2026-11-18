# Dependence measures

## Objective

Dependence measures describe statistical relationships between random
variables. For continuous variables, **mutual information** (MI) measures how
far the joint distribution differs from the distribution that would arise under
independence.

The `pycopula_ml.dependence` module uses fitted **bivariate copulas** to
estimate MI and to construct pairwise dependence matrices.

This is a new layer in the package's statistical workflow:

```text
Original observations (X, Y)
            ↓
Marginal CDFs or rank-based pseudo-observations (U, V)
            ↓
Copula parameter estimation (theta_hat)
            ↓
Fitted copula C_theta_hat
            ↓
Model-based mutual information I_C(X; Y)
```

## Mutual information

For two jointly continuous random variables, with a well-defined density and
finite mutual information,

$$
I(X;Y)
=\int_{\mathbb{R}}\!\int_{\mathbb{R}}
f_{XY}(x,y)
\log\left(\frac{f_{XY}(x,y)}{f_X(x)f_Y(y)}\right)
\,dx\,dy.
$$

It can also be described as a Kullback--Leibler divergence:

$$
I(X;Y)=D_{\mathrm{KL}}(P_{XY}\,\|\,P_X\otimes P_Y).
$$

Consequently, MI is nonnegative, is zero exactly under independence, and is
symmetric in the two variables. It is **not** a correlation coefficient and
its magnitude is not a percentage. Natural logarithms give values in **nats**.

When the joint distribution is represented by a copula density
$c(u,v)$, Sklar's theorem yields

$$
\boxed{
I(X;Y)=\int_0^1\!\int_0^1
c(u,v)\log c(u,v)\,du\,dv
}.
$$

This formulation explains why MI can be obtained directly from a fitted
copula without specifying the marginal density formulas again. See
[Copula-based mutual information](../dependence/mutual_information.md).

## Pairwise dependence in a multivariate sample

Let $X\in\mathbb R^{n\times d}$ contain $n$ observations and $d$ features.
After fitting a separate bivariate copula for every pair of features, the
package can construct:

- $\Theta\in\mathbb R^{d\times d}$, the fitted copula parameters;
- $L\in\mathbb R^{d\times d}$, maximized copula log-likelihoods;
- $M\in\mathbb R^{d\times d}$, model-based mutual information in nats.

These are **feature-by-feature matrices**, not sample-by-sample data matrices.
The current implementation uses zero diagonals as placeholders; this does not
represent the theoretical MI of a continuous variable with itself.

For supervised classification, the matrices must be estimated **separately
within each class** from training observations belonging to that class. This
is analogous to the class-specific dependence learning in R package
MLCOPULA 1.1.1; construction of its full probabilistic classifier is a
separate stage.

See [Pairwise dependence matrices](../dependence/weight_matrices.md) and the
[Dependence API](../api/dependence.md).

## Scope

Pairwise MI provides a useful edge-weight candidate for graph construction.
However, it does not by itself encode conditional independence, identify
causal relationships, build a maximum-spanning tree, or train a graph neural
network. Those operations require additional models and algorithms.
