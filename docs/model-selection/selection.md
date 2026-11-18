# Copula model selection

The `model_selection` package exposes three distinct responsibilities:

1. `aic(log_likelihood, n_parameters)`: Akaike's information criterion.
2. `bic(log_likelihood, n_parameters, n_observations)`: Bayesian information criterion.
3. `copula_selector(candidates, u, v, criterion="bic")`: estimate and rank
   several bivariate copula families on identical paired observations.

For $n$ paired observations and $k$ fitted parameters,

$$
\mathrm{AIC} = 2k - 2\ell(\hat\theta),\qquad
\mathrm{BIC} = k\ln(n) - 2\ell(\hat\theta).
$$

```python
from pycopula_ml.copulas import FrankCopula, GaussianCopula
from pycopula_ml.model_selection import copula_selector

candidate_copulas = {
    FrankCopula: (-8.0, 8.0),
    GaussianCopula: (-0.98, 0.98),
}
result = copula_selector(candidate_copulas, u, v, criterion="bic")
print(result.selected.name, result.selected.theta)
```

The vectors `u` and `v` must represent the **same observations** across
candidate models. Do not use test-set observations for fitting or ranking.
`CopulaSelectionResult.candidates` contains ordered `CopulaCriteria` values
including family, theta, log-likelihood, AIC and BIC.

The implementation is intentionally separate from marginal selection and the
MI-driven graph constructor. For tests, see
`tests/model_selection/test_criteria.py` and `test_selector.py`.
