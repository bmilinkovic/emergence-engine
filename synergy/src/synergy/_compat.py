"""Compatibility helpers for MATLAB-style PhiID calculations."""

from __future__ import annotations

import numpy as np


def is_discrete(x: np.ndarray) -> bool:
    """Classify discrete input for integer-like arrays."""
    if np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.bool_):
        return True
    return np.all(np.isfinite(x)) and np.all(np.equal(x, np.round(x)))


def ensure_combined(u: np.ndarray) -> np.ndarray:
    """Collapse possibly-multivariate rows into a single discrete variable.

    MATLAB equivalent:
      [~,~,V] = unique(U', 'rows'); V = V(:)' - 1;
    """
    u = np.asarray(u)
    if u.ndim == 1:
        return u.astype(int)
    if u.shape[0] == 1:
        return u[0].astype(int)

    rows = np.ascontiguousarray(u.T)
    _, inv = np.unique(rows, axis=0, return_inverse=True)
    return inv.astype(int)


def zscore_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit standard deviation, for stable covariance calculations."""
    x = np.asarray(x, dtype=float)
    std = np.std(x, axis=1, ddof=0)
    std = np.where(std < eps, 1.0, std)
    return x / std[:, None]
