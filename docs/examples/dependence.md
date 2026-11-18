# Dependence examples

These examples use the `pycopula_ml.copulas`,
`pycopula_ml.estimation`, and `pycopula_ml.dependence` modules.

## Example 1: from two measurements to mutual information

This example follows
$X,Y\rightarrow U,V\rightarrow\widehat\theta\rightarrow I(X;Y)$.

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas.parametrics.frank import FrankCopula
from pycopula_ml.dependence import mutual_information
from pycopula_ml.estimation import fit_copula_mle

x = np.array([10, 15, 20, 25, 30], dtype=float)
y = np.array([20, 35, 30, 45, 50], dtype=float)

n = len(x)
u = rankdata(x, method="average") / (n + 1)
v = rankdata(y, method="average") / (n + 1)

fit = fit_copula_mle(
    FrankCopula,
    u,
    v,
    bounds=((-20.0, -1e-4), (1e-4, 20.0)),
)

copula = FrankCopula(theta=fit.theta)
mi = mutual_information(copula)

print(f"theta: {fit.theta:.6f}")
print(f"log-likelihood: {fit.log_likelihood:.6f}")
print(f"mutual information: {mi:.6f} nats")
```

The five observations illustrate the API only; they should not be interpreted
as an adequately powered model fit.

## Example 2: Gaussian closed-form mutual information

```python
from pycopula_ml.dependence import gaussian_mutual_information

for rho in (0.0, 0.3, -0.6, 0.6):
    print(rho, gaussian_mutual_information(rho))
```

The values at `rho=-0.6` and `rho=0.6` are equal. MI captures the amount of
dependence represented by the fitted Gaussian copula, not its direction.

## Example 3: class-specific matrices with four features

The script below generates three classes with different dependencies between
features `X1` and `X2`. For each class it obtains separate $4\times4$ matrices
for `theta`, log-likelihood, and mutual information.

```python
import numpy as np
from scipy.stats import rankdata

from pycopula_ml.copulas.parametrics.gaussian import GaussianCopula
from pycopula_ml.dependence import build_weight_matrices

rng = np.random.default_rng(2026)
n_per_class = 120

samples = []
labels = []
for label, rho in enumerate((0.1, 0.6, -0.65)):
    z = rng.standard_normal((n_per_class, 4))
    z[:, 1] = rho * z[:, 0] + np.sqrt(1.0 - rho**2) * z[:, 1]
    samples.append(z)
    labels.extend([label] * n_per_class)

x_train = np.vstack(samples)
y_train = np.asarray(labels)

results_by_class = {}
for label in np.unique(y_train):
    x_class = x_train[y_train == label]
    u_class = rankdata(x_class, axis=0, method="average") / (len(x_class) + 1)

    result = build_weight_matrices(
        u_class,
        GaussianCopula,
        bounds=(-0.95, 0.95),
        weight_method="mutual_information",
        feature_names=("X1", "X2", "X3", "X4"),
    )
    results_by_class[label] = result

    print(f"\nClass {label}")
    print("theta:\n", np.round(result.theta, 4))
    print("log-likelihood:\n", np.round(result.log_likelihood, 4))
    print("MI (nats):\n", np.round(result.mutual_information, 4))
```

The estimated matrices are symmetric with zero diagonals. In this simulation,
classes with larger absolute pairwise correlations should generally have
higher MI for the `(X1, X2)` pair, despite differing signs.

## Example 4: display MI as a labeled matrix

The following fragment continues **Example 3**:

```python
import pandas as pd

result = results_by_class[0]
mi_table = pd.DataFrame(
    result.mutual_information,
    index=result.feature_names,
    columns=result.feature_names,
)
print(mi_table.round(4))
```

Pandas is only required for tabular display, not for the dependence
calculations.

## Validation

Run the companion dependence tests with:

```bash
python -m pytest tests/dependence -v
```

See [Dependence validation](../testing/dependence_validation.md) for the test
coverage and recommended statistical checks.
