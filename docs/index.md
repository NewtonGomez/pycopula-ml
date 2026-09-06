# pycopula-ml documentation

This directory documents the current architecture, mathematical foundations, implemented API, testing strategy, and development roadmap of **pycopula-ml**.

The project currently concentrates on the two transformations required before a copula-based classifier can be constructed:

$$
X
\rightarrow
U
$$

through marginal modeling, and

$$
(U,V)
\rightarrow
C_\theta(u,v)
$$

through bivariate copulas.

## Contents

### Getting Started

- [Development workflow](getting-started/development.md)

### Concepts

- [Mathematical foundations](concepts/mathematical_foundations.md)
- [Copulas](concepts/copulas.md)

### Copulas

- [Frank copula](copulas/frank_copula.md)


### API Reference

- [Copula API](api/copulas.md)

### References

- [References](references/references.md)
