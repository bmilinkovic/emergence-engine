"""Modality-independent summaries of subject-level Phi_ID matrices."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from .metrics import weighted_global_efficiency, weighted_modularity


@dataclass(frozen=True)
class SubjectRecord:
    """One participant's matrices and optional comparison networks.

    ``atoms`` maps estimator names (such as ``mmi``) to matrices named
    ``rtr`` and ``sts``. ``connectivity`` can contain any matched square
    matrices, for example functional or structural connectivity.
    """
    subject_id: str
    group: str
    atoms: Mapping[str, Mapping[str, np.ndarray]]
    connectivity: Mapping[str, np.ndarray] | None = None


@dataclass
class AnalysisResult:
    """Tables and metadata consumed by statistics and plotting functions."""
    subject: pd.DataFrame
    similarity: pd.DataFrame
    nodal: pd.DataFrame
    systems: pd.DataFrame
    records: list[SubjectRecord]
    node_labels: tuple[str, ...]
    node_systems: tuple[str, ...] | None
    group_order: tuple[str, ...]
    reference_group: str
    methods: tuple[str, ...]


def _square(matrix: np.ndarray, n: int, label: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"{label} must have shape ({n}, {n}); got {arr.shape}")
    return arr


def _regional_mean(matrix: np.ndarray) -> np.ndarray:
    values = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(values, np.nan)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    return np.divide(np.nansum(values, axis=1), count,
                     out=np.full(len(values), np.nan), where=count > 0)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    tri = np.triu_indices_from(a, 1)
    x, y = a[tri], b[tri]
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.std(x[good]) < 1e-12 or np.std(y[good]) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x[good], y[good]).statistic)


def analyse_subjects(
    records: Sequence[SubjectRecord], *, node_labels: Sequence[str] | None = None,
    node_systems: Sequence[str] | None = None, group_order: Sequence[str] | None = None,
    reference_group: str | None = None,
) -> AnalysisResult:
    """Summarise arbitrary groups and node sets without preprocessing assumptions.

    Matrices must share a node order. Missing edges may be NaN; all finite
    undirected off-diagonal edges contribute to edge-level summaries.
    """
    records = list(records)
    if not records:
        raise ValueError("At least one SubjectRecord is required")
    if len({r.subject_id for r in records}) != len(records):
        raise ValueError("subject_id values must be unique")
    first_methods = tuple(records[0].atoms)
    if not first_methods or any(set(r.atoms) != set(first_methods) for r in records):
        raise ValueError("Every subject needs the same nonempty estimator set")
    first = np.asarray(records[0].atoms[first_methods[0]]["rtr"])
    if first.ndim != 2 or first.shape[0] != first.shape[1] or first.shape[0] < 2:
        raise ValueError("Atom matrices must be square with at least two nodes")
    n = first.shape[0]
    labels = tuple(node_labels) if node_labels is not None else tuple(f"node_{i+1}" for i in range(n))
    if len(labels) != n or len(set(labels)) != n:
        raise ValueError("node_labels must be unique and match matrix size")
    systems = tuple(node_systems) if node_systems is not None else None
    if systems is not None and len(systems) != n:
        raise ValueError("node_systems must match matrix size")
    groups = tuple(group_order) if group_order is not None else tuple(dict.fromkeys(r.group for r in records))
    if len(set(groups)) != len(groups) or set(groups) != {r.group for r in records}:
        raise ValueError("group_order must contain each observed group exactly once")
    reference = reference_group or groups[0]
    if reference not in groups:
        raise ValueError("reference_group must be one of group_order")
    tri = np.triu_indices(n, 1)
    subject_rows: list[dict] = []
    similarity_rows: list[dict] = []
    nodal_rows: list[dict] = []
    system_rows: list[dict] = []
    for record in records:
        for method in first_methods:
            atom_set = record.atoms[method]
            if not {"rtr", "sts"} <= set(atom_set):
                raise ValueError(f"{record.subject_id}/{method} needs rtr and sts")
            rtr = _square(atom_set["rtr"], n, "rtr")
            sts = _square(atom_set["sts"], n, "sts")
            region = {"rtr": _regional_mean(rtr), "sts": _regional_mean(sts)}
            valid = np.isfinite(region["rtr"]) & np.isfinite(region["sts"])
            gradient = np.full(n, np.nan)
            if valid.sum() > 1:
                gradient[valid] = (stats.rankdata(region["sts"][valid])
                                   - stats.rankdata(region["rtr"][valid]))
            values_by_metric = {
                "rtr": region["rtr"], "sts": region["sts"],
                "balance": region["sts"] - region["rtr"],
                "rank_gradient": gradient,
            }
            for metric, values in values_by_metric.items():
                for i, value in enumerate(values):
                    nodal_rows.append(dict(subject_id=record.subject_id, group=record.group,
                        method=method, metric=metric, node_index=i, node_label=labels[i], value=float(value)))
                if systems is not None:
                    for system in dict.fromkeys(systems):
                        idx = np.array([s == system for s in systems])
                        finite = values[idx & np.isfinite(values)]
                        system_rows.append(dict(subject_id=record.subject_id, group=record.group,
                            method=method, metric=metric, system=system,
                            value=float(finite.mean()) if len(finite) else np.nan))
            for atom, matrix in (("rtr", rtr), ("sts", sts), ("balance", sts-rtr)):
                edges = matrix[tri]
                finite = edges[np.isfinite(edges)]
                positive = np.where(np.isfinite(matrix) & (matrix > 0), matrix, 0.0)
                np.fill_diagonal(positive, 0.0)
                subject_rows.append(dict(subject_id=record.subject_id, group=record.group,
                    method=method, atom=atom, n_valid_edges=len(finite),
                    mean_edge=float(finite.mean()) if len(finite) else np.nan,
                    positive_fraction=float(np.mean(finite > 0)) if len(finite) else np.nan,
                    global_efficiency=float(weighted_global_efficiency(positive)),
                    modularity=float(weighted_modularity(positive)) if np.any(positive) else 0.0))
                if record.connectivity:
                    for name, ref in record.connectivity.items():
                        reference_matrix = _square(ref, n, name)
                        similarity_rows.append(dict(subject_id=record.subject_id,
                            group=record.group, method=method, atom=atom,
                            reference=name, rho=_rho(reference_matrix, matrix)))
    return AnalysisResult(pd.DataFrame(subject_rows), pd.DataFrame(similarity_rows),
                          pd.DataFrame(nodal_rows), pd.DataFrame(system_rows),
                          records, labels, systems, groups, reference, first_methods)


# Keep the earlier public name available for existing notebooks and scripts.
analyze_subjects = analyse_subjects
