"""Subject-level tests copied from the original DoC analysis."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    ok = np.isfinite(arr)
    if not np.any(ok):
        return out
    vals = arr[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = ranked.size
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    back = np.empty_like(q)
    back[order] = q
    out[ok] = back
    return out


def _perm_test_paired(a: np.ndarray, b: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        return float("nan"), float("nan")
    diff = y - x
    obs = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size), replace=True)
    perm = (signs * diff[None, :]).mean(axis=1)
    p = (1.0 + np.sum(np.abs(perm) >= abs(obs))) / (n_perm + 1.0)
    return obs, float(p)


def _perm_test_one_sample(x: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, arr.size), replace=True)
    perm = (signs * arr[None, :]).mean(axis=1)
    p = (1.0 + np.sum(np.abs(perm) >= abs(obs))) / (n_perm + 1.0)
    return obs, float(p)


def _subject_similarity_tests(
    subject_df: pd.DataFrame,
    *,
    measure_name: str,
    rtr_col: str,
    sts_col: str,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    i = 0
    for method, sub_method in subject_df.groupby("method", sort=False):
        for cohort, sub_cohort in sub_method.groupby("cohort", sort=False):
            a = sub_cohort[rtr_col].to_numpy(dtype=float)
            b = sub_cohort[sts_col].to_numpy(dtype=float)
            diff, p = _perm_test_paired(a, b, n_perm=n_perm, seed=seed + i)
            rows.append(
                {
                    "measure": measure_name,
                    "method": method,
                    "cohort": cohort,
                    "n_subjects": int(np.sum(np.isfinite(a) & np.isfinite(b))),
                    "mean_rtr": float(np.nanmean(a)),
                    "mean_sts": float(np.nanmean(b)),
                    "mean_diff_sts_minus_rtr": diff,
                    "perm_p": p,
                }
            )
            i += 1
    out = pd.DataFrame(rows)
    out["perm_q_fdr"] = _bh_fdr(out["perm_p"].to_numpy(dtype=float))
    out["significant_fdr"] = out["perm_q_fdr"] < 0.05
    return out


def _subject_graph_tests(
    graph_df: pd.DataFrame,
    *,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["global_efficiency", "modularity"]
    i = 0
    for method, sub_method in graph_df.groupby("method", sort=False):
        for cohort, sub_cohort in sub_method.groupby("cohort", sort=False):
            piv = sub_cohort.pivot_table(index="subject_id", columns="atom", values=metrics)
            for metric in metrics:
                a = piv[(metric, "rtr")].to_numpy(dtype=float)
                b = piv[(metric, "sts")].to_numpy(dtype=float)
                diff, p = _perm_test_paired(a, b, n_perm=n_perm, seed=seed + i)
                rows.append(
                    {
                        "measure": metric,
                        "method": method,
                        "cohort": cohort,
                        "n_subjects": int(np.sum(np.isfinite(a) & np.isfinite(b))),
                        "mean_rtr": float(np.nanmean(a)),
                        "mean_sts": float(np.nanmean(b)),
                        "mean_diff_sts_minus_rtr": diff,
                        "perm_p": p,
                    }
                )
                i += 1
    out = pd.DataFrame(rows)
    out["perm_q_fdr"] = _bh_fdr(out["perm_p"].to_numpy(dtype=float))
    out["significant_fdr"] = out["perm_q_fdr"] < 0.05
    return out


def _surface_gradient_roi_tests(
    gradient_df: pd.DataFrame,
    *,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    sub = gradient_df.loc[gradient_df["include_in_cortical_only"] == 1].copy()
    rows: list[dict[str, Any]] = []
    i = 0
    for (method, cohort, roi_index, roi_label), g in sub.groupby(
        ["method", "cohort", "roi_index", "roi_label"], sort=False
    ):
        vals = g["gradient_value"].to_numpy(dtype=float)
        mean_val, p = _perm_test_one_sample(vals, n_perm=n_perm, seed=seed + i)
        rows.append(
            {
                "method": method,
                "cohort": cohort,
                "roi_index": int(roi_index),
                "roi_label": str(roi_label),
                "n_subjects": int(np.isfinite(vals).sum()),
                "mean_gradient": mean_val,
                "perm_p": p,
            }
        )
        i += 1
    out = pd.DataFrame(rows)
    out["perm_q_fdr"] = _bh_fdr(out["perm_p"].to_numpy(dtype=float))
    out["significant_fdr"] = out["perm_q_fdr"] < 0.05
    return out
