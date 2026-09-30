# Parametric bivariate copulas

`pycopula_lm.copulas.parametrics` contains one-parameter bivariate copula
families with a common API:

```python
copula.cdf(u, v)
copula.pdf(u, v)
copula.logpdf(u, v)
copula.log_likelihood(u, v)
```

The public CDF, PDF, and log-PDF methods are implemented by
`ParametricBivariateCopula`. Each family supplies its own `_cdf`, `_pdf`,
`_logpdf`, and parameter validation.

## Independence

The base class treats the family-specific independence parameter explicitly:

\[
C(u,v)=uv,\qquad c(u,v)=1,\qquad \log c(u,v)=0.
\]

This avoids numerical indeterminacies in family formulas at independence.

| Family | Parameter domain used by the library | Independence |
| --- | --- | --- |
| Frank | \(\theta\in\mathbb{R}\) | \(\theta=0\) |
| Clayton | \(\theta>-1\) | \(\theta=0\) |
| Gaussian | \(-1<\theta<1\) | \(\theta=0\) |
| Joe | \(\theta\ge1\) | \(\theta=1\) |
| Gumbel-Hougaard | \(\theta\ge1\) | \(\theta=1\) |
| Ali-Mikhail-Haq | \(-1\le\theta<1\) | \(\theta=0\) |

For Clayton, the singular endpoint \(\theta=-1\) is intentionally excluded
because this package interface assumes an ordinary continuous bivariate
density.

## Importing

```python
from pycopula_lm.copulas import (
    AliMikhailHaqCopula,
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    JoeCopula,
)
```
