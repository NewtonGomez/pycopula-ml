# Graph-construction tests

Run applicable test modules from the repository root:

```bash
python -m pytest tests/graphs -v
python -m pytest --cov=pycopula_ml --cov-report=term-missing
```

## Expected properties

- A connected spanning tree or chain over all feature vertices.
- Exactly `d - 1` selected undirected edges.
- No duplicate vertices, cycles, or invalid edge indices.
- Deterministic tie-breaking and no in-place mutation of input matrices.
- Input validation for symmetry, finiteness, nonnegativity, and diagonals.
- Correct handling of zero-weight edges and feature names.
- Agreement between edge indices, attributes, and adjacency matrices.
- Exact MST score against small hand-checkable examples.
- Chain scores verified without claiming global optimality.

For end-to-end checks, compute pseudo-observations and pairwise copula mutual
information before constructing a graph. Verify that all fitting uses the
training partition only.
