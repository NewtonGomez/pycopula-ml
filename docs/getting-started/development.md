# Development workflow

## Environment

Create a virtual environment:

```bash
python3 -m venv .env
source .env/bin/activate
```

Install in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Prefer:

```bash
python -m pip
python -m pytest
```

over bare `pip3` or `pytest` commands when multiple Python installations exist
on the same machine.

## Editable installation

The project uses a `src/` layout:

```text
src/
└── pycopula_ml/
    ├── copulas/
    ├── estimation/
    └── marginals/
```

Editable installation ensures imports resolve through the active environment
while source changes remain immediately available.

## Style

The project uses Ruff for formatting and linting.

```bash
ruff format .
ruff check .
```

The current configuration targets PEP 8-compatible conventions and import
ordering.

## Test before commit

Recommended sequence:

```bash
ruff format .
ruff check .
python -m pytest
```

For the estimation module specifically:

```bash
python -m pytest tests/estimation -v
```

## Test layers

The project distinguishes several testing responsibilities.

### Unit tests

Unit tests are deterministic and belong in `tests/`.

For estimation, they should test the optimizer independently from a specific
copula family whenever possible.

For example, a synthetic log-likelihood with a known maximum can verify that
`fit_copula_mle()` correctly:

- finds the optimum;
- searches multiple intervals;
- validates input domains;
- rejects invalid bounds.

### Integration tests

Integration tests verify that multiple package layers work together.

A representative fitting path is:

```text
bivariate_pseudo_observations()
        ↓
FrankCopula
        ↓
fit_copula_mle()
```

These tests should remain deterministic if they are included in continuous
integration.

### Statistical validation scripts

Monte Carlo experiments should live under a path such as:

```text
scripts/validation/
```

They are used to study statistical properties such as:

- estimator bias;
- MAE;
- RMSE;
- variability across repeated samples;
- effect of sample size.

They should not replace deterministic unit tests.

## Distribution vs import name

The package distribution name is:

```text
pycopula-ml
```

The Python import name is:

```text
pycopula_ml
```

Example:

```python
import pycopula_ml
```

## Current dependencies

Core:

- NumPy
- SciPy

Development:

- pytest
- pytest-cov
- Ruff

Documentation dependencies may include:

- Sphinx
- numpydoc

SciPy currently provides both:

- continuous distribution functionality used by marginal models;
- bounded scalar numerical optimization used internally by the estimation
  layer.

User code should normally access optimization through `pycopula_ml.estimation`
rather than import `scipy.optimize` directly.

## Public APIs

Prefer imports from package `__init__.py` files.

Recommended:

```python
from pycopula_ml.copulas import FrankCopula
from pycopula_ml.estimation import fit_copula_mle
from pycopula_ml.marginals import MarginalSelector
```

rather than importing internal implementation paths unless needed for
development.

Recommended:

```python
from pycopula_ml.estimation import fit_copula_mle
```

Internal-development import:

```python
from pycopula_ml.estimation.maximum_likelihood import fit_copula_mle
```

The first form is preferred for user-facing code.

## Internal helpers

Functions and methods beginning with `_`, such as:

```text
_prepare_inputs()
_prepare_pseudo_observations()
_negative_log_likelihood()
```

are internal implementation details.

They are not considered stable public API.

## Documentation conventions

Docstrings currently follow NumPy-style structure:

```text
Parameters
----------
...

Returns
-------
...

Raises
------
...

Notes
-----
...
```

Mathematical expressions are written with Sphinx-compatible math directives
where appropriate.

## Version status

Current development version:

```text
0.1.0
```

Development status:

```text
Alpha
```

The public API should not yet be considered stable.
