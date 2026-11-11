"""Continuous (Gaussian) PhiID computation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from ._compat import zscore_rows
from .partition import minimum_information_partition
from .redundancy import (
    double_redundancy_mmi_gaussian,
    local_entropy_gaussian,
    redundancy_ccs,
    redundancy_mmi,
)

ATOM_NAMES = [
    "rtr",
    "rtx",
    "rty",
    "rts",
    "xtr",
    "xtx",
    "xty",
    "xts",
    "ytr",
    "ytx",
    "yty",
    "yts",
    "str",
    "stx",
    "sty",
    "sts",
]


# Shared by local and accelerated pairwise implementations.
PHIID_LATTICE = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )

@dataclass(frozen=True)
class PhiIDResult:
    atoms: dict[str, float]
    local_atoms: dict[str, np.ndarray] | None
    partition: tuple[tuple[int, ...], tuple[int, ...]] | None = None


def phiid_full(
    *args,
    tau: int = 1,
    measure: str = "CCS",
    partition: tuple[Iterable[int], Iterable[int]] | None = None,
    return_local: bool = False,
) -> PhiIDResult:
    """Compute Gaussian PhiID.

    Supported forms:
    - `phiid_full(X, tau=1, measure='CCS', partition=None)` where X is [D, T]
    - `phiid_full(X1, X2, Y1, Y2, measure='CCS')`

    Notes:
    - If no partition is supplied, a normalized effective-
      information MIB is selected by exhaustive search.
    - `measure` currently supports `CCS` and `MMI`.
    """
    if len(args) == 1:
        x = np.asarray(args[0], dtype=float)
        return _td_phiid(
            x,
            tau=tau,
            measure=measure,
            partition=partition,
            return_local=return_local,
        )
    if len(args) == 4:
        return _four_vector_phiid(
            np.asarray(args[0], dtype=float),
            np.asarray(args[1], dtype=float),
            np.asarray(args[2], dtype=float),
            np.asarray(args[3], dtype=float),
            measure=measure,
            return_local=return_local,
        )
    raise ValueError("phiid_full expects either 1 or 4 positional arrays.")


def _td_phiid(
    x: np.ndarray,
    tau: int,
    measure: str,
    partition: tuple[Iterable[int], Iterable[int]] | None,
    return_local: bool,
) -> PhiIDResult:
    if x.ndim != 2:
        raise ValueError("Input must be 2D [variables, time].")
    d, t = x.shape
    if not isinstance(tau, (int, np.integer)) or isinstance(tau, (bool, np.bool_)) or tau < 1:
        raise ValueError("tau must be a positive integer.")
    if d < 2 or t - tau <= d:
        raise ValueError("Need at least two variables and more lagged samples than variables.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input must contain only finite values.")

    sx = zscore_rows(x)
    p1, p2 = (minimum_information_partition(sx, tau=tau, kind="gaussian")
              if partition is None else _validate_partition(partition, d))

    result = _four_vector_phiid(
        sx[p1, : t - tau],
        sx[p2, : t - tau],
        sx[p1, tau:],
        sx[p2, tau:],
        measure=measure,
        return_local=return_local,
    )
    return replace(result, partition=(tuple(p1), tuple(p2)))


def _four_vector_phiid(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    measure: str,
    return_local: bool,
) -> PhiIDResult:
    x1, x2, y1, y2 = (_as_matrix(v) for v in (x1, x2, y1, y2))
    t = x1.shape[1]
    if x2.shape[1] != t or y1.shape[1] != t or y2.shape[1] != t:
        raise ValueError("All inputs must have equal time length.")
    if t <= sum(v.shape[0] for v in (x1, x2, y1, y2)):
        raise ValueError("Need more samples than the joint variable dimension.")

    measure_key = measure.lower()
    if measure_key not in {"ccs", "mmi"}:
        raise NotImplementedError(
            "Only 'CCS' and 'MMI' are currently implemented in this Python port."
        )

    x = np.vstack([x1, x2, y1, y2])
    sx = zscore_rows(x)

    p1 = np.arange(0, x1.shape[0])
    p2 = np.arange(p1[-1] + 1, p1[-1] + 1 + x2.shape[0])
    t1 = np.arange(p2[-1] + 1, p2[-1] + 1 + y1.shape[0])
    t2 = np.arange(t1[-1] + 1, t1[-1] + 1 + y2.shape[0])

    def h(idx: np.ndarray) -> np.ndarray:
        return local_entropy_gaussian(sx[idx, :])

    h_p1 = h(p1)
    h_p2 = h(p2)
    h_t1 = h(t1)
    h_t2 = h(t2)

    h_p1p2 = h(np.r_[p1, p2])
    h_t1t2 = h(np.r_[t1, t2])
    h_p1t1 = h(np.r_[p1, t1])
    h_p1t2 = h(np.r_[p1, t2])
    h_p2t1 = h(np.r_[p2, t1])
    h_p2t2 = h(np.r_[p2, t2])

    h_p1p2t1 = h(np.r_[p1, p2, t1])
    h_p1p2t2 = h(np.r_[p1, p2, t2])
    h_p1t1t2 = h(np.r_[p1, t1, t2])
    h_p2t1t2 = h(np.r_[p2, t1, t2])

    h_p1p2t1t2 = h(np.r_[p1, p2, t1, t2])

    ixytab = h_p1p2 + h_t1t2 - h_p1p2t1t2

    ixta = h_p1 + h_t1 - h_p1t1
    ixtb = h_p1 + h_t2 - h_p1t2
    iyta = h_p2 + h_t1 - h_p2t1
    iytb = h_p2 + h_t2 - h_p2t2

    ixyta = h_p1p2 + h_t1 - h_p1p2t1
    ixytb = h_p1p2 + h_t2 - h_p1p2t2
    ixtab = h_p1 + h_t1t2 - h_p1t1t2
    iytab = h_p2 + h_t1t2 - h_p2t1t2

    red_fun = redundancy_ccs if measure_key == "ccs" else lambda a, b, c: redundancy_mmi(a, b)

    rxyta = red_fun(ixta, iyta, ixyta)
    rxytb = red_fun(ixtb, iytb, ixytb)
    rxytab = red_fun(ixtab, iytab, ixytab)
    rabtx = red_fun(ixta, ixtb, ixtab)
    rabty = red_fun(iyta, iytb, iytab)
    rabtxy = red_fun(ixyta, ixytb, ixytab)

    if measure_key == "mmi":
        rtr = double_redundancy_mmi_gaussian(
            sx[p1, :],
            sx[p2, :],
            sx[t1, :],
            sx[t2, :],
        )
    else:
        double_coinfo = (
            -ixta
            - ixtb
            - iyta
            - iytb
            + ixtab
            + iytab
            + ixyta
            + ixytb
            - ixytab
            + rxyta
            + rxytb
            - rxytab
            + rabtx
            + rabty
            - rabtxy
        )
        signs = np.column_stack([np.sign(ixta), np.sign(ixtb), np.sign(iyta), np.sign(iytb), np.sign(double_coinfo)])
        same = np.all(signs == signs[:, [0]], axis=1)
        rtr = same * double_coinfo

    reds = np.vstack(
        [
            rtr,
            rxyta,
            rxytb,
            rxytab,
            rabtx,
            rabty,
            rabtxy,
            ixta,
            ixtb,
            iyta,
            iytb,
            ixyta,
            ixytb,
            ixtab,
            iytab,
            ixytab,
        ]
    )

    m = PHIID_LATTICE

    partials = np.linalg.solve(m, reds)
    local = {k: partials[i, :] for i, k in enumerate(ATOM_NAMES)}
    means = {k: float(np.mean(v)) for k, v in local.items()}

    return PhiIDResult(atoms=means, local_atoms=local if return_local else None)


def _as_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[np.newaxis, :]
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("Each variable must be a nonempty 1D or 2D array.")
    if not np.all(np.isfinite(value)):
        raise ValueError("Inputs must contain only finite values.")
    return value


def _validate_partition(partition, dimension: int) -> tuple[list[int], list[int]]:
    if partition is None:
        if dimension == 2:
            return [0], [1]
        raise ValueError("For more than two variables, provide partition=(indices_a, indices_b).")
    try:
        first, second = (list(side) for side in partition)
    except (TypeError, ValueError) as exc:
        raise ValueError("partition must contain two nonempty index sequences.") from exc
    all_indices = first + second
    if (not first or not second or
        any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) for i in all_indices) or
        sorted(all_indices) != list(range(dimension))):
        raise ValueError("partition must split every variable index exactly once.")
    return first, second
