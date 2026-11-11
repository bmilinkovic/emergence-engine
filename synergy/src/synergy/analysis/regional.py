"""Regional values copied from the original DoC analysis."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
COHORTS=("control","emcs","mcs","uws","coma")
METHODS=("mmi","ccs")
U=np.triu_indices(90,1)

def regional_mean(matrix: np.ndarray) -> np.ndarray:
    values = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(values, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=1)


def gradient(sts: np.ndarray, rtr: np.ndarray) -> np.ndarray:
    a, b = regional_mean(sts), regional_mean(rtr)
    good = np.isfinite(a) & np.isfinite(b)
    result = np.full(90, np.nan)
    if good.sum() > 1:
        result[good] = stats.rankdata(a[good]) - stats.rankdata(b[good])
    return result


def subject_values(out: Path, names: list[str]):
    qc = pd.read_csv(out / "edge_validity_qc.csv")
    nodal, global_rows = [], []
    matrices = {}
    for row in qc.itertuples(index=False):
        with np.load(out / "subject_matrices" / f"{row.subject_id}.npz") as z:
            for method in METHODS:
                sts = np.asarray(z[f"{method}_sts"], float)
                rtr = np.asarray(z[f"{method}_rtr"], float)
                balance = sts - rtr
                matrices.setdefault((method, row.cohort, "sts"), []).append(sts)
                matrices.setdefault((method, row.cohort, "rtr"), []).append(rtr)
                for metric, values in (("rank_gradient", gradient(sts, rtr)),
                                       ("balance", regional_mean(balance)),
                                       ("sts", regional_mean(sts)),
                                       ("rtr", regional_mean(rtr))):
                    for roi, value in enumerate(values):
                        nodal.append(dict(method=method, cohort=row.cohort,
                                          subject_id=row.subject_id, metric=metric,
                                          roi_index=roi+1, roi_label=names[roi],
                                          value=float(value)))
                for atom, matrix in (("sts", sts), ("rtr", rtr),
                                     ("balance", balance)):
                    global_rows.append(dict(method=method, cohort=row.cohort,
                                            subject_id=row.subject_id, atom=atom,
                                            mean_edge_value=float(np.nanmean(matrix[U]))))
    return pd.DataFrame(nodal), pd.DataFrame(global_rows), matrices
