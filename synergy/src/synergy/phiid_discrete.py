"""Discrete PhiID computation."""

from __future__ import annotations

from typing import Iterable
from dataclasses import replace

import numpy as np

from ._compat import ensure_combined, is_discrete
from .phiid import ATOM_NAMES, PhiIDResult, _validate_partition
from .partition import minimum_information_partition
from .redundancy import double_redundancy_mmi_discrete, local_mi_discrete, redundancy_ccs, redundancy_mmi


def phiid_full_discrete(
    *args,
    tau: int = 1,
    measure: str = "MMI",
    partition: tuple[Iterable[int], Iterable[int]] | None = None,
    return_local: bool = False,
) -> PhiIDResult:
    """Compute discrete PhiID for either TD-series or explicit 4-vector inputs.

    Supported measures: `MMI`, `CCS`, and Varley's shared-exclusions `SX`.
    `SX` requires integer-coded discrete states and returns values in bits;
    continuous data must be discretized explicitly before calling it.
    """
    if len(args) == 1:
        x = np.asarray(args[0])
        return _td_phiid_discrete(
            x,
            tau=tau,
            measure=measure,
            partition=partition,
            return_local=return_local,
        )
    if len(args) == 4:
        return _four_vector_discrete(
            np.asarray(args[0]),
            np.asarray(args[1]),
            np.asarray(args[2]),
            np.asarray(args[3]),
            measure=measure,
            return_local=return_local,
        )
    raise ValueError("phiid_full_discrete expects either 1 or 4 positional arrays.")


def _binarify(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v)
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
        raise ValueError("Discrete inputs must be numeric or Boolean.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Discrete inputs must contain only finite values.")
    if is_discrete(arr):
        return arr
    if arr.ndim == 1:
        thr = np.mean(arr)
        return (arr > thr).astype(int)
    thr = np.mean(arr, axis=1, keepdims=True)
    return (arr > thr).astype(int)


def _td_phiid_discrete(
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
    if d < 2 or t - tau < 2:
        raise ValueError("Need at least two variables and two lagged samples.")
    if measure.lower() == "sx" and not is_discrete(x):
        raise ValueError("SX requires discrete states; discretize continuous data explicitly.")

    bx = _binarify(x)
    p1, p2 = (minimum_information_partition(bx, tau=tau, kind="discrete")
              if partition is None else _validate_partition(partition, d))

    result = _four_vector_discrete(
        bx[p1, : t - tau],
        bx[p2, : t - tau],
        bx[p1, tau:],
        bx[p2, tau:],
        measure=measure,
        return_local=return_local,
    )
    return replace(result, partition=(tuple(p1), tuple(p2)))


def _as_single_discrete_var(v: np.ndarray) -> np.ndarray:
    vv = _binarify(v)
    if vv.ndim == 1:
        vv = vv[np.newaxis, :]
    return ensure_combined(vv)


def _four_vector_discrete(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    measure: str,
    return_local: bool,
) -> PhiIDResult:
    for value in (x1, x2, y1, y2):
        if np.asarray(value).ndim not in (1, 2) or np.asarray(value).size == 0:
            raise ValueError("Each variable must be a nonempty 1D or 2D array.")
    t = x1.shape[-1]
    if x2.shape[-1] != t or y1.shape[-1] != t or y2.shape[-1] != t:
        raise ValueError("All input arrays must have matching time dimension.")
    if t < 2:
        raise ValueError("Need at least two samples.")

    measure_key = measure.lower()
    if measure_key not in {"mmi", "ccs", "sx"}:
        raise NotImplementedError("Only 'MMI', 'CCS', and 'SX' are implemented in this Python port.")
    if measure_key == "sx" and any(not is_discrete(np.asarray(value)) for value in (x1, x2, y1, y2)):
        raise ValueError("SX requires discrete states; discretize continuous data explicitly.")

    b_x = np.vstack(
        [
            _as_single_discrete_var(x1),
            _as_single_discrete_var(x2),
            _as_single_discrete_var(y1),
            _as_single_discrete_var(y2),
        ]
    )

    if measure_key == "sx":
        from .shared_exclusions import phiid_shared_exclusions

        return phiid_shared_exclusions(b_x, return_local=return_local)

    ixta = local_mi_discrete(b_x, [0], [2])
    ixtb = local_mi_discrete(b_x, [0], [3])
    iyta = local_mi_discrete(b_x, [1], [2])
    iytb = local_mi_discrete(b_x, [1], [3])

    ixyta = local_mi_discrete(b_x, [0, 1], [2])
    ixytb = local_mi_discrete(b_x, [0, 1], [3])
    ixtab = local_mi_discrete(b_x, [0], [2, 3])
    iytab = local_mi_discrete(b_x, [1], [2, 3])
    ixytab = local_mi_discrete(b_x, [0, 1], [2, 3])

    red_fun = redundancy_ccs if measure_key == "ccs" else lambda a, b, c: redundancy_mmi(a, b)

    rxyta = red_fun(ixta, iyta, ixyta)
    rxytb = red_fun(ixtb, iytb, ixytb)
    rxytab = red_fun(ixtab, iytab, ixytab)
    rabtx = red_fun(ixta, ixtb, ixtab)
    rabty = red_fun(iyta, iytb, iytab)
    rabtxy = red_fun(ixyta, ixytb, ixytab)

    if measure_key == "mmi":
        rtr = double_redundancy_mmi_discrete(b_x)
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

    m = np.array(
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

    partials = np.linalg.solve(m, reds)
    local = {k: partials[i, :] for i, k in enumerate(ATOM_NAMES)}
    means = {k: float(np.mean(v)) for k, v in local.items()}

    return PhiIDResult(atoms=means, local_atoms=local if return_local else None)
