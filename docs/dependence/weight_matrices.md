# Pairwise copula weight matrices

## Purpose

`build_weight_matrices()` fits one bivariate copula to **each distinct pair
of feature columns** in a matrix of pseudo-observations and returns three
pairwise matrices:

- $L$: maximized sample copula log-likelihood;
- $\Theta$: estimated dependence parameter $\widehat\theta$;
- $M$: model-based mutual information.

## Input and dimensions

Let $U\in(0,1)^{n\times d}$ have one row per observation and one column per
feature. The function returns arrays in $\mathbb R^{d\times d}$.

For a pair $i<j$:

$$
\widehat\theta_{ij}
=\underset{\theta}{\arg\max}
\sum_{k=1}^{n}
\log c_\theta(u_{ki},u_{kj}),
$$

$$
L_{ij}
=\sum_{k=1}^{n}
\log c_{\widehat\theta_{ij}}(u_{ki},u_{kj}),
$$

$$
M_{ij}
=\int_0^1\!\int_0^1
c_{\widehat\theta_{ij}}(u,v)
\log c_{\widehat\theta_{ij}}(u,v)\,du\,dv.
$$

For example, with four features, the fitted MI matrix has the form

$$
M=\begin{pmatrix}
0 & I_{12} & I_{13} & I_{14}\\
I_{12} & 0 & I_{23} & I_{24}\\
I_{13} & I_{23} & 0 & I_{34}\\
I_{14} & I_{24} & I_{34} & 0
\end{pmatrix}.
$$

There are $d(d-1)/2$ distinct pairs. The implementation copies each value
into its transposed position so the three matrices are symmetric, and sets
their diagonals to zero.

!!! important "The diagonal is a software convention"
    The zeros on the diagonal mean "not estimated" or "no self-edge". They
    do **not** assert that $I(X_i;X_i)=0$, which is false for a nondegenerate
    continuous variable with itself. The current weight builder assumes
    symmetric bivariate copula families for pairwise modeling.

## Pairwise matrix example

The function expects **pseudo-observations**, not untransformed features.
Create them before calling `build_weight_matrices()`:

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas.parametrics.gaussian import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices

rng = np.random.default_rng(2026)
x = rng.normal(size=(120, 4))

u = rankdata(x, axis=0, method="average") / (len(x) + 1)

result = build_weight_matrices(
    u,
    GaussianCopula,
    bounds=(-0.95, 0.95),
    feature_names=("X1", "X2", "X3", "X4"),
    weight_method="mutual_information",
)

print("theta:\n", result.theta)
print("log-likelihood:\n", result.log_likelihood)
print("mutual information (nats):\n", result.mutual_information)
print("selected weights:\n", result.weights)
```

`bounds` is passed to the package's existing `fit_copula_mle()` estimator.
The valid domain depends on the copula class; for Frank, its estimation
interval is often split around independence. Refer to
[Maximum-likelihood estimation](../estimation/maximum_likelihood.md).

## One matrix set per class

For supervised learning, let $y\in\{c_1,\ldots,c_K\}^n$ contain training
class labels. For each class $c$, restrict the **training** observations to
that class, form pseudo-observations within it, and fit the pairwise copulas:

$$
X^{(c)}=\{x_k:y_k=c\},
\qquad
U^{(c)}=\operatorname{pobs}(X^{(c)}),
$$

$$
\bigl(L^{(c)},\Theta^{(c)},M^{(c)}\bigr)
=\operatorname{BuildMatrices}(U^{(c)}).
$$

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas.parametrics.gaussian import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices

# x_train: array of shape (n_samples, n_features)
# y_train: labels for the same training rows
results_by_class = {}

for label in np.unique(y_train):
    x_class = x_train[y_train == label]
    u_class = rankdata(x_class, axis=0, method="average") / (len(x_class) + 1)

    results_by_class[label] = build_weight_matrices(
        u_class,
        GaussianCopula,
        bounds=(-0.95, 0.95),
        weight_method="mutual_information",
    )
```

This procedure creates **one set of feature-by-feature matrices for each
class**. The class label itself is not a column of $U^{(c)}$ and does not
form a pair with each feature.

!!! warning "Prevent data leakage"
    For training/validation/test splits, estimate all marginal
    transformations, copulas, parameters, and dependence matrices **from
    training data only**. Keep preprocessing inside each cross-validation
    fold to avoid leakage between training and evaluation partitions.

## Weight selection

The builder exposes `weight_method`:

| Option | Returned `result.weights` | Interpretation |
| --- | --- | --- |
| `"mutual_information"` | `result.mutual_information` | Model-based dependence intensity, in nats |
| `"likelihood"` | `result.log_likelihood` | Maximized copula log-likelihood on the fitting pairs |

The two criteria are different. Because log-likelihood is a sample sum, its
scale depends on class sample size; avoid treating values from different
classes as directly comparable evidence without appropriate normalization
or a probabilistic model.

The current builder computes all three matrices regardless of the chosen
weight method. Numerical MI integration may therefore add computational cost.

## Result object

The function returns `CopulaWeightMatrices`, with fields:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `theta` | `ndarray[d, d]` | Fitted parameter for every feature pair |
| `log_likelihood` | `ndarray[d, d]` | Maximized copula log-likelihoods |
| `mutual_information` | `ndarray[d, d]` | Copula-based MI values (nats) |
| `weights` | `ndarray[d, d]` | Selected matrix, according to `weight_method` |
| `feature_names` | `tuple[str, ...]` | Labels of feature columns |
| `weight_method` | `str` | Selected weighting criterion |

The result dataclass is frozen, although its underlying NumPy arrays are not
necessarily immutable. An input `pandas.DataFrame` can provide column names
when `feature_names` is not explicitly given.

## Implementation details and limits

The function processes every distinct pair exactly once, calls
`fit_copula_mle()`, constructs the copula at the fitted `theta`, evaluates
`mutual_information()`, and fills the three matrices. Data must be a finite,
two-dimensional array with at least two rows and columns, with values strictly
inside $(0,1)$.

The package currently does **not** implement in this function:

- automatic selection of a different copula family per feature pair;
- maximum-spanning tree or chain extraction;
- class-conditional likelihood for new observations;
- model training and class prediction.

The resulting matrices can be passed to graph construction, but they are
not a complete classifier.

See the [Dependence API](../api/dependence.md),
[Python examples](../examples/dependence.md), and
[Dependence validation](../testing/dependence_validation.md).
