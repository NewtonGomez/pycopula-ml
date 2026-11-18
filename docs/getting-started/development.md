# Development workflow

## Environment

Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The distribution name is `pycopula-ml`; the Python import name is
`pycopula_ml`. The package uses the `src/` layout.

## Python style: PEP 8

Use PEP 8 conventions for Python source code and documentation examples:

- Four spaces per indentation level; no tab indentation.
- `snake_case` for modules, variables, and functions; `PascalCase` for classes.
- Consistent imports: standard library, third-party packages, local package.
- Prefer lines of 79 characters or fewer in docstrings and comments; keep
  code within the repository's configured formatter line length.
- Add type hints where they clarify public interfaces.
- Use NumPy-style docstrings for statistical functions and classes.

Run the project's formatter and linter:

```bash
ruff format .
ruff check .
```

Formatting documentation examples does not change the library's source code.

## Tests

Run unit and integration tests:

```bash
python -m pytest
python -m pytest --cov=pycopula_ml --cov-report=term-missing
```

The repository's test paths may include `tests/estimation`,
`tests/dependence`, `tests/graphs`, and `tests/integration`; verify their
presence before running path-specific selections.

Deterministic unit tests should cover numerical domains, input validation,
independence parameters, estimation results, graph invariants, and error paths.
Monte Carlo experiments can supplement these tests but should not replace them.

## Public API

Prefer imports exposed by package `__init__.py` modules:

```python
from pycopula_ml.copulas import FrankCopula
from pycopula_ml.estimation import fit_copula_mle
from pycopula_ml.marginals import MarginalSelector
```

Internal functions beginning with `_` are implementation details and should
not be treated as stable public APIs.

## Documentation

Documentation pages are Markdown files in `docs/`, with site configuration
in `mkdocs.yml`. Build and preview the site with:

```bash
python -m pip install mkdocs-material
python -m mkdocs build --strict
python -m mkdocs serve
```

Use descriptive Markdown headings, relative links, fenced code blocks with
language identifiers, and mathematical notation consistent with the other
pages. Keep examples small, reproducible, and application-neutral.

## Dependencies and scope

NumPy and SciPy are the documented statistical dependencies. Optional packages
used only in presentation examples are not required by the core APIs. Verify
optional dependency groups against the repository's `pyproject.toml` before
advertising installation extras.
