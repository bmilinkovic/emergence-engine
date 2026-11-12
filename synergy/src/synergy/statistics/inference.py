"""Group-independent inference for subject, node, and system summaries."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from ..analysis.workflow import AnalysisResult


@dataclass
class InferenceResult:
    """Inference tables with family-wise FDR columns."""
    paired_atoms: pd.DataFrame
    paired_similarity: pd.DataFrame
    group_subject: pd.DataFrame
    group_nodal: pd.DataFrame
    group_systems: pd.DataFrame
    one_sample_systems: pd.DataFrame


def bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjustment, retaining missing values."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    good = np.isfinite(p)
    if good.any():
        order = np.argsort(p[good])
        ranked = p[good][order] * good.sum() / np.arange(1, good.sum()+1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        values = np.empty(good.sum())
        values[order] = np.clip(ranked, 0, 1)
        out[good] = values
    return out


def _paired(x, y, rng: np.random.Generator, n_perm: int) -> tuple[int, float, float]:
    a, b = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(a) & np.isfinite(b)
    delta = b[valid] - a[valid]
    if not len(delta):
        return 0, np.nan, np.nan
    observed = float(delta.mean())
    count = 0
    # Bounded batches avoid allocating n_perm × n_subjects for large studies.
    remaining = n_perm
    while remaining:
        batch = min(remaining, 1024)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(delta)))
        count += int(np.count_nonzero(np.abs((signs*delta).mean(axis=1)) >= abs(observed)-1e-14))
        remaining -= batch
    return len(delta), observed, (count+1)/(n_perm+1)


def _paired_table(frame: pd.DataFrame, value_col: str, index_cols: list[str],
                  n_perm: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(index_cols, sort=False):
        pivot = group.pivot_table(index="subject_id", columns="atom", values=value_col, aggfunc="first")
        if not {"rtr", "sts"} <= set(pivot):
            continue
        n, effect, p = _paired(pivot["rtr"], pivot["sts"], rng, n_perm)
        row = dict(zip(index_cols, key if isinstance(key, tuple) else (key,)))
        rows.append({**row, "n_subjects": n, "sts_minus_rtr": effect, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_fdr"] = bh_fdr(out.p)
    return out


def _contrasts(frame: pd.DataFrame, *, reference: str, value_col: str,
               keys: list[str], groups: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for key, sub in frame.groupby(keys, sort=False):
        fields = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        base = sub.loc[sub.group.eq(reference), value_col].dropna().to_numpy(float)
        for group in groups:
            if group == reference:
                continue
            other = sub.loc[sub.group.eq(group), value_col].dropna().to_numpy(float)
            if len(base) > 1 and len(other) > 1:
                effect = float(other.mean()-base.mean())
                p = float(stats.ttest_ind(other, base, equal_var=False).pvalue)
            else:
                effect, p = np.nan, np.nan
            rows.append({**fields, "group": group, "reference_group": reference,
                         "n_reference": len(base), "n_group": len(other),
                         "group_minus_reference": effect, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        # Each estimator and metric/atom is one FDR family across nodes and groups.
        family = [c for c in ("method", "metric", "atom") if c in out]
        out["q_fdr"] = out.groupby(family, sort=False).p.transform(
            lambda p: bh_fdr(p.to_numpy(float)))
    return out


def infer_groups(result: AnalysisResult, *, n_perm: int = 10_000,
                 seed: int = 0) -> InferenceResult:
    """Test within-subject atoms and each group against a chosen reference.

    Paired atom and similarity effects use sign-flip permutations. Between-group
    effects use Welch tests. The reference group is set in ``analyse_subjects``.
    """
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    rng = np.random.default_rng(seed)
    pairs = []
    for measure in ("mean_edge", "global_efficiency", "modularity"):
        table = _paired_table(result.subject, measure, ["method", "group"], n_perm, rng)
        if len(table):
            table.insert(0, "measure", measure)
            pairs.append(table)
    paired_atoms = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    if len(paired_atoms):
        paired_atoms["q_fdr"] = paired_atoms.groupby("measure").p.transform(
            lambda p: bh_fdr(p.to_numpy(float)))
    paired_similarity = (_paired_table(result.similarity.rename(columns={"rho": "value"}),
                       "value", ["reference", "method", "group"], n_perm, rng)
                         if len(result.similarity) else pd.DataFrame())
    group_subject = _contrasts(result.subject, reference=result.reference_group,
        value_col="mean_edge", keys=["method", "atom"], groups=result.group_order)
    group_nodal = _contrasts(result.nodal, reference=result.reference_group,
        value_col="value", keys=["method", "metric", "node_index", "node_label"],
        groups=result.group_order)
    group_systems = (_contrasts(result.systems, reference=result.reference_group,
        value_col="value", keys=["method", "metric", "system"], groups=result.group_order)
        if len(result.systems) else pd.DataFrame())
    one_rows = []
    if len(result.systems):
        systems = result.systems.loc[result.systems.metric.eq("rank_gradient")]
        for key, sub in systems.groupby(["method", "group", "system"], sort=False):
            vals = sub.value.dropna().to_numpy(float)
            n, mean, p = _paired(np.zeros(len(vals)), vals, rng, n_perm)
            one_rows.append(dict(zip(("method", "group", "system"), key)) |
                            {"n_subjects": n, "mean_gradient": mean, "p": p})
    one_sample = pd.DataFrame(one_rows)
    if len(one_sample):
        one_sample["q_fdr"] = one_sample.groupby("method").p.transform(
            lambda p: bh_fdr(p.to_numpy(float)))
    return InferenceResult(paired_atoms, paired_similarity, group_subject,
                           group_nodal, group_systems, one_sample)
