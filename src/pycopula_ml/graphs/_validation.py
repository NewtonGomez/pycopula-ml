"""Shared validation of copula mutual-information graph inputs.

Keep graph topology algorithms independent of the statistical estimation layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_mi_matrix(
    matrix: ArrayLike,
    names: Sequence[str] | None,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Validate symmetric, nonnegative MI values and optional feature names."""
    try:
        mi = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("mi_matrix must contain numeric values.") from exc
    if mi.ndim != 2 or mi.shape[0] != mi.shape[1] or mi.shape[0] < 2:
        raise ValueError("mi_matrix must be square with at least two features.")
    if not np.isfinite(mi).all():
        raise ValueError("mi_matrix must contain only finite values.")
    if not np.allclose(mi, mi.T, rtol=1e-10, atol=1e-10):
        raise ValueError("mi_matrix must be symmetric.")
    if np.any(mi < -1e-10):
        raise ValueError("Mutual information cannot be negative.")
    if not np.allclose(np.diag(mi), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("mi_matrix must have a zero diagonal (no self-edges).")

    n = mi.shape[0]
    if names is None:
        labels = tuple(f"X{i + 1}" for i in range(n))
    else:
        labels = tuple(names)
        if len(labels) != n or any(not isinstance(x, str) or not x for x in labels):
            raise ValueError(
                "feature_names must be nonempty strings matching the matrix."
            )
        if len(set(labels)) != n:
            raise ValueError("feature_names must be unique.")
    cleaned = np.maximum(0.0, (mi + mi.T) / 2.0)
    np.fill_diagonal(cleaned, 0.0)
    return cleaned, labels
