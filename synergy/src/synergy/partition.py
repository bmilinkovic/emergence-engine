"""Exhaustive minimum-information bipartition selection.

The Gaussian and discrete selectors scan
bipartitions and minimize effective information divided by the smaller
past-state entropy. Effective information is time-delayed system MI minus
the sum of within-part time-delayed MIs. This module performs the same
calculation without a JVM. Entropy and MI units cancel in the score.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ._compat import ensure_combined


def _partitions(dimension: int):
    # Exactly one member of each complementary pair of nonempty subsets.
    # Put variable 0 on the first side to make the orientation deterministic.
    indices = tuple(range(1, dimension))
    for size in range(dimension - 1):
        for extra in combinations(indices, size):
            left = (0,) + extra
            right = tuple(i for i in range(dimension) if i not in left)
            yield left, right


def _gaussian_entropy(covariance: np.ndarray, indices: tuple[int, ...]) -> float:
    block = covariance[np.ix_(indices, indices)]
    sign, logdet = np.linalg.slogdet(block)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("Lagged covariance is singular or not positive definite.")
    return 0.5 * (len(indices) * np.log(2 * np.pi * np.e) + logdet)


def _gaussian_mi(covariance: np.ndarray, left: tuple[int, ...],
                 right: tuple[int, ...]) -> float:
    return (_gaussian_entropy(covariance, left)
            + _gaussian_entropy(covariance, right)
            - _gaussian_entropy(covariance, left + right))


def _discrete_entropy(data: np.ndarray) -> float:
    _, counts = np.unique(ensure_combined(data), return_counts=True)
    probability = counts / counts.sum()
    return float(-np.dot(probability, np.log(probability)))


def _discrete_mi(past: np.ndarray, future: np.ndarray) -> float:
    return (_discrete_entropy(past) + _discrete_entropy(future)
            - _discrete_entropy(np.vstack((past, future))))


def minimum_information_partition(
    data: np.ndarray, *, tau: int = 1, kind: str = "gaussian",
    max_variables: int = 12,
) -> tuple[list[int], list[int]]:
    """Find the normalized effective-information minimum bipartition.

    ``data`` is variables × time; ``tau`` is a positive sample lag. The
    Gaussian version fits a full lagged sample covariance. The discrete
    version treats each integer-valued row as a categorical variable.
    Exhaustive search visits ``2**(D-1)-1`` bipartitions; ``max_variables``
    guards against accidental exponential work. A zero smaller-part entropy
    cannot normalize the score and raises an error.
    """
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError("data must have shape [variables, time].")
    dimension, length = x.shape
    if not isinstance(tau, (int, np.integer)) or isinstance(tau, (bool, np.bool_)) or tau < 1:
        raise ValueError("tau must be a positive integer.")
    if dimension < 2 or dimension > max_variables:
        raise ValueError(f"Variable count must be between 2 and {max_variables} for exhaustive search.")
    if length - tau <= dimension:
        raise ValueError("Need more lagged samples than variables.")
    if not np.issubdtype(x.dtype, np.number) and x.dtype != np.bool_:
        raise ValueError("data must be numeric or Boolean.")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must contain only finite values.")
    if kind not in {"gaussian", "discrete"}:
        raise ValueError("kind must be 'gaussian' or 'discrete'.")

    past, future = x[:, :-tau], x[:, tau:]
    if kind == "gaussian":
        covariance = np.cov(np.vstack((past, future)))
        all_variables = tuple(range(dimension))
        total = _gaussian_mi(covariance, all_variables,
                             tuple(i + dimension for i in all_variables))
        def part_mi(indices):
            return _gaussian_mi(covariance, indices,
                                tuple(i + dimension for i in indices))
        def part_entropy(indices):
            return _gaussian_entropy(covariance, indices)
    else:
        if not np.all(np.equal(x, np.round(x))):
            raise ValueError("Discrete partition selection requires integer-valued data.")
        total = _discrete_mi(past, future)
        def part_mi(indices):
            return _discrete_mi(past[list(indices)], future[list(indices)])
        def part_entropy(indices):
            return _discrete_entropy(past[list(indices)])

    best = None
    best_score = np.inf
    for left, right in _partitions(dimension):
        norm = min(part_entropy(left), part_entropy(right))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("A partition has zero or invalid past-state entropy; MIB is undefined.")
        score = (total - part_mi(left) - part_mi(right)) / norm
        if score < best_score:
            best_score = score
            best = (list(left), list(right))
    assert best is not None
    return best
