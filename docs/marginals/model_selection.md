# Marginal model selection

## `MarginalSelector`

`MarginalSelector` fits several candidate parametric distributions and ranks successful fits.

Default candidates currently include:

- Normal;
- Gamma;
- Lognormal;
- Weibull;
- Exponential.

## Selection criteria

Supported criteria:

```text
aic
bic
ks
```

The selected model minimizes the configured quantity.

### AIC

$$
AIC=2k-2\ell.
$$

Lower is better.

### BIC

$$
BIC=k\log n-2\ell.
$$

Lower is better.

### KS statistic

$$
D
=
\sup_x
\left|
F_n(x)-F(x)
\right|.
$$

Lower is better.

## Why the KS p-value is not the default selector

The classical one-sample KS p-value assumes a fully specified theoretical distribution.

When the same sample is used both to estimate distribution parameters and to compute the KS test, the standard p-value is no longer formally calibrated in the usual way.

The package therefore stores it as an exploratory diagnostic rather than using it as the default selection rule.

## Selected does not mean true

The package uses the name

```python
selected_model_
```

rather than interpreting the model as the true generating distribution.

The meaning is:

> preferred model among the supplied candidate set under the configured selection criterion.

Aliases such as `best_model_` may exist for convenience, but `selected_model_` is statistically more precise.

## Example

```python
from pycopula_ml.marginals import MarginalSelector


selector = MarginalSelector(
    criterion="aic",
)

selector.fit(x)

print(selector.summary())

model = selector.selected_model_
u = selector.transform(x)
```

## Custom candidates

```python
from scipy.stats import gamma, lognorm, norm

from pycopula_ml.marginals import (
    MarginalCandidate,
    MarginalSelector,
)


candidates = [
    MarginalCandidate(
        name="normal",
        distribution=norm,
    ),
    MarginalCandidate(
        name="gamma",
        distribution=gamma,
        fit_kwargs={"floc": 0.0},
    ),
    MarginalCandidate(
        name="lognormal",
        distribution=lognorm,
        fit_kwargs={"floc": 0.0},
    ),
]

selector = MarginalSelector(
    candidates=candidates,
    criterion="aic",
)
```

## Warnings

The selector can warn about:

- small samples;
- fitted support boundaries close to observed extrema;
- candidate fitting failures.

These warnings are diagnostic. They do not automatically invalidate a model.

## Small samples

Automatic distribution selection is especially unstable when the sample is very small relative to the number of fitted parameters.

The package should therefore report rather than hide this uncertainty.

## Future improvements

Planned selection improvements include:

- AICc;
- goodness-of-fit bootstrap;
- diagnostic plots;
- transformed-uniform diagnostics;
- cross-validation where appropriate;
- automatic support-aware candidate filtering.
