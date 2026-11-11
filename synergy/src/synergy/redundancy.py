"""Redundancy and local information helper functions for PhiID/PID."""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

from ._compat import ensure_combined


def _safe_cov(x: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    c = np.cov(x)
    if c.ndim == 0:
        c = np.array([[float(c)]])
    c = np.asarray(c, dtype=float)
    c = c + ridge * np.eye(c.shape[0])
    return c


def local_entropy_gaussian(x: np.ndarray) -> np.ndarray:
    """Local entropy for matrix x shaped [variables, time]."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]

    mu = np.mean(x, axis=1)
    cov = _safe_cov(x)
    # Evaluate each timepoint sample under fitted Gaussian.
    pts = x.T
    logpdf = multivariate_normal.logpdf(pts, mean=mu, cov=cov, allow_singular=False)
    return -np.asarray(logpdf)


def local_mi_gaussian(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    if src.ndim == 1:
        src = src[np.newaxis, :]
    if tgt.ndim == 1:
        tgt = tgt[np.newaxis, :]

    h_src = local_entropy_gaussian(src)
    h_tgt = local_entropy_gaussian(tgt)
    h_joint = local_entropy_gaussian(np.vstack([src, tgt]))
    return h_src + h_tgt - h_joint


def local_mi_discrete(x: np.ndarray, src: list[int] | np.ndarray, tgt: list[int] | np.ndarray) -> np.ndarray:
    """Plug-in local MI using natural log, consistent with PhiID atom algebra."""
    x = np.asarray(x)
    src_var = ensure_combined(x[np.array(src), :])
    tgt_var = ensure_combined(x[np.array(tgt), :])

    joint = np.stack([src_var, tgt_var], axis=1)
    _, inv_joint, cnt_joint = np.unique(joint, axis=0, return_inverse=True, return_counts=True)
    _, inv_src, cnt_src = np.unique(src_var, return_inverse=True, return_counts=True)
    _, inv_tgt, cnt_tgt = np.unique(tgt_var, return_inverse=True, return_counts=True)

    n = float(src_var.size)
    p_joint = cnt_joint[inv_joint] / n
    p_src = cnt_src[inv_src] / n
    p_tgt = cnt_tgt[inv_tgt] / n

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p_joint / (p_src * p_tgt))


def redundancy_mmi(mi1: np.ndarray, mi2: np.ndarray) -> np.ndarray:
    return mi1 if np.nanmean(mi1) < np.nanmean(mi2) else mi2


def redundancy_ccs(mi1: np.ndarray, mi2: np.ndarray, mi12: np.ndarray) -> np.ndarray:
    c = mi12 - mi1 - mi2
    signs = np.column_stack([np.sign(mi1), np.sign(mi2), np.sign(mi12), np.sign(-c)])
    same = np.all(signs == signs[:, [0]], axis=1)
    return same * (-c)


def double_redundancy_mmi_gaussian(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
) -> np.ndarray:
    src = [x1, x2]
    tgt = [y1, y2]
    vals = [local_mi_gaussian(s, t) for s in src for t in tgt]
    means = [np.nanmean(v) for v in vals]
    return vals[int(np.argmin(means))]


def double_redundancy_mmi_discrete(
    x: np.ndarray,
) -> np.ndarray:
    # x is expected to be [4 vars, T] after ensure_combined handling.
    vals = [
        local_mi_discrete(x, [0], [2]),
        local_mi_discrete(x, [0], [3]),
        local_mi_discrete(x, [1], [2]),
        local_mi_discrete(x, [1], [3]),
    ]
    means = [np.nanmean(v) for v in vals]
    return vals[int(np.argmin(means))]
