# Dependence testing and validation

## Unit tests

Where present in the repository, tests in `tests/dependence/` should cover:

1. Exact Gaussian mutual information, including independence and sign
   symmetry.
2. Domain validation for Gaussian correlation parameters.
3. Numerical integration of copula-based mutual information.
4. Zero mutual information under independence.
5. Shape, symmetry, and diagonal conventions of pairwise matrices.
6. Invalid pseudo-observations, nonfinite values, and fitting failures.

```bash
python -m pytest tests/dependence -v
```

## Statistical checks

- Check family-specific mutual information at valid reference parameters.
- Compare quadrature orders (for example, `64`, `128`, and `256`) and report
  convergence differences when dependence is strong.
- Separate estimation error from numerical integration error.
- Test tied ranks, constant features, and insufficient class sample sizes.
- Confirm that class-specific transformations do not access validation or
  test data.
- Keep random seeds fixed in reproducible simulations.

Numerical quadrature can differ slightly from reference values even when
the underlying copula parameter is identical. Record tolerances and floating-
point assumptions in statistical regression tests.
