"""Figures for arbitrary groups, estimators, nodes, and modalities."""
from __future__ import annotations
from pathlib import Path
import re
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..analysis.workflow import AnalysisResult
from ..statistics.inference import InferenceResult

COLOURS = {"rtr": "#2E5EAA", "sts": "#D43D2A", "balance": "#8A5A44"}
COLORS = COLOURS  # Backward compatible public name.


def _groups(result: AnalysisResult, order: Sequence[str] | None) -> tuple[str, ...]:
    groups = tuple(order) if order is not None else result.group_order
    if set(groups) != set(result.group_order) or len(groups) != len(result.group_order):
        raise ValueError("group_order must contain every observed group exactly once")
    return groups


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "figure"


def save_figure(fig: plt.Figure, output: Path, *, formats=("png", "pdf", "svg")) -> list[Path]:
    """Save a figure under one stem in the requested formats."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output.with_suffix("." + fmt)
        fig.savefig(path, dpi=260 if fmt == "png" else None, bbox_inches="tight")
        paths.append(path)
    return paths


def plot_atom_pairs(result: AnalysisResult, *, measure: str = "mean_edge",
                    reference: str | None = None, inference: InferenceResult | None = None,
                    group_order: Sequence[str] | None = None) -> plt.Figure:
    """Show paired RTR and STS values for any subject-level measure."""
    groups = _groups(result, group_order)
    if reference is None:
        if measure not in result.subject:
            raise ValueError(f"Unknown subject measure: {measure}")
        frame = result.subject
        value_col = measure
        tests = inference.paired_atoms if inference is not None else pd.DataFrame()
        test_filter = lambda f: f.loc[f.measure.eq(measure)] if "measure" in f else f
    else:
        frame = result.similarity.loc[result.similarity.reference.eq(reference)]
        if frame.empty:
            raise ValueError(f"No connectivity reference named {reference!r}")
        value_col = "rho"
        tests = inference.paired_similarity if inference is not None else pd.DataFrame()
        test_filter = lambda f: f.loc[f.reference.eq(reference)] if "reference" in f else f
    fig, axes = plt.subplots(len(result.methods), 1, squeeze=False,
                            figsize=(max(7, 1.6*len(groups)+2), 3.8*len(result.methods)),
                            constrained_layout=True)
    for row, method in enumerate(result.methods):
        ax = axes[row, 0]
        subset = frame.loc[frame.method.eq(method) & frame.atom.isin(("rtr", "sts"))]
        for i, group in enumerate(groups):
            for atom, offset in (("rtr", -0.18), ("sts", 0.18)):
                values = subset.loc[subset.group.eq(group) & subset.atom.eq(atom), value_col].dropna().to_numpy(float)
                if not len(values):
                    continue
                x = i + offset
                ax.scatter(np.full(len(values), x), values, color=COLOURS[atom], alpha=0.35, s=12)
                ax.errorbar(x, values.mean(), yerr=(values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0),
                            fmt="o", color=COLOURS[atom], markeredgecolor="white", ms=9, capsize=3, lw=2)
            if len(tests):
                row_test = test_filter(tests)
                row_test = row_test.loc[row_test.method.eq(method) & row_test.group.eq(group)]
                if len(row_test) and np.isfinite(row_test.q_fdr.iloc[0]) and row_test.q_fdr.iloc[0] < .05:
                    ax.text(i, 1.02, "*", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=15)
        ax.set_title(method.upper(), loc="left", weight="bold")
        ax.set_xticks(range(len(groups)), groups)
        ax.set_ylabel("Spearman ρ" if reference is not None else measure.replace("_", " "))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLOURS[a], label=a.upper()) for a in ("rtr", "sts")]
    fig.legend(handles=handles, loc="upper right")
    return fig


def plot_global_atoms(result: AnalysisResult, *, group_order: Sequence[str] | None = None) -> plt.Figure:
    """Violin distributions of global RTR, STS, and balance by group."""
    groups = _groups(result, group_order)
    atoms = ("sts", "rtr", "balance")
    fig, axes = plt.subplots(len(result.methods), len(atoms), squeeze=False,
                            figsize=(max(10, 1.15*len(groups)*3), 3.6*len(result.methods)),
                            constrained_layout=True)
    for row, method in enumerate(result.methods):
        for col, atom in enumerate(atoms):
            ax = axes[row, col]
            sub = result.subject.loc[result.subject.method.eq(method) & result.subject.atom.eq(atom)]
            arrays = [sub.loc[sub.group.eq(group), "mean_edge"].dropna().to_numpy(float) for group in groups]
            for i, values in enumerate(arrays):
                if not len(values):
                    continue
                if len(values) > 1:
                    violin = ax.violinplot([values], positions=[i], widths=.65, showextrema=False)
                    for body in violin["bodies"]:
                        body.set_facecolor(COLOURS[atom]); body.set_alpha(.2)
                ax.scatter(np.full(len(values), i), values, s=12, alpha=.45, color=COLOURS[atom])
                ax.plot([i-.14, i+.14], [np.median(values)]*2, color="#263238", lw=2)
            ax.set_title(f"{method.upper()} · {atom.upper()}")
            ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0: ax.set_ylabel("Mean edge value (nats)")
    return fig


def plot_system_gradients(result: AnalysisResult, *, group_order: Sequence[str] | None = None) -> plt.Figure:
    """Gradient violins for user-provided node systems; no atlas is assumed."""
    if result.node_systems is None or result.systems.empty:
        raise ValueError("Pass node_systems to analyse_subjects for this figure")
    groups = _groups(result, group_order)
    systems = tuple(dict.fromkeys(result.node_systems))
    fig, axes = plt.subplots(len(result.methods), len(groups), squeeze=False,
                            figsize=(max(5, 2.5*len(groups)), 3.6*len(result.methods)),
                            sharey=True, constrained_layout=True)
    for row, method in enumerate(result.methods):
        for col, group in enumerate(groups):
            ax = axes[row, col]
            sub = result.systems.loc[result.systems.method.eq(method) & result.systems.group.eq(group)
                                     & result.systems.metric.eq("rank_gradient")]
            for i, system in enumerate(systems):
                values = sub.loc[sub.system.eq(system), "value"].dropna().to_numpy(float)
                if len(values) > 1:
                    violin = ax.violinplot([values], positions=[i], widths=.7, showextrema=False)
                    for body in violin["bodies"]:
                        body.set_facecolor("#4C78A8"); body.set_alpha(.35)
                if len(values): ax.scatter(np.full(len(values), i), values, s=8, alpha=.35, color="#2E4057")
            ax.axhline(0, color="#777", lw=.8)
            ax.set_title(f"{method.upper()} · {group}")
            ax.set_xticks(range(len(systems)), systems, rotation=65, ha="right", fontsize=8)
            if col == 0: ax.set_ylabel("Synergy − redundancy rank")
            ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_node_gradients(result: AnalysisResult, *, group_order: Sequence[str] | None = None) -> plt.Figure:
    """Heatmaps of mean nodal gradients for arbitrary regions or EEG channels."""
    groups = _groups(result, group_order)
    frame = result.nodal.loc[result.nodal.metric.eq("rank_gradient")]
    fig, axes = plt.subplots(len(result.methods), 1, squeeze=False,
                            figsize=(max(8, .35*len(result.node_labels)+3), 2.0*len(result.methods)+1),
                            constrained_layout=True)
    finite = np.abs(frame.value.to_numpy(float))
    finite = finite[np.isfinite(finite)]
    limit = max(1, float(finite.max())) if len(finite) else 1.0
    for row, method in enumerate(result.methods):
        ax = axes[row, 0]
        matrix = np.full((len(groups), len(result.node_labels)), np.nan)
        for i, group in enumerate(groups):
            sub = frame.loc[frame.method.eq(method) & frame.group.eq(group)]
            for j, values in sub.groupby("node_index"):
                matrix[i, int(j)] = values.value.mean()
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_yticks(range(len(groups)), groups)
        ax.set_xticks(range(len(result.node_labels)), result.node_labels, rotation=75, ha="right", fontsize=8)
        ax.set_title(f"{method.upper()} · mean node gradient", loc="left")
        fig.colorbar(im, ax=ax, shrink=.7, label="Rank difference")
    return fig


def plot_regional_effects(result: AnalysisResult, inference: InferenceResult, *,
                          method: str, metric: str = "balance") -> plt.Figure:
    """Heatmap of group-minus-reference effects for every node."""
    if method not in result.methods:
        raise ValueError(f"Unknown estimator: {method}")
    groups = tuple(g for g in result.group_order if g != result.reference_group)
    if not groups:
        raise ValueError("At least two groups are needed for contrasts")
    frame = inference.group_nodal
    sub = frame.loc[frame.method.eq(method) & frame.metric.eq(metric)]
    if sub.empty:
        raise ValueError(f"No regional test for {method}/{metric}")
    matrix = np.full((len(groups), len(result.node_labels)), np.nan)
    q = np.full_like(matrix, np.nan)
    for row in sub.itertuples(index=False):
        i, j = groups.index(row.group), int(row.node_index)
        matrix[i, j], q[i, j] = row.group_minus_reference, row.q_fdr
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(1e-9, float(finite.max())) if len(finite) else 1.0
    fig, ax = plt.subplots(figsize=(max(8, .35*len(result.node_labels)+3), max(2.5, .75*len(groups)+1)),
                           constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    for i, j in zip(*np.where(q < .05)):
        ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=12)
    ax.set_yticks(range(len(groups)), groups)
    ax.set_xticks(range(len(result.node_labels)), result.node_labels, rotation=75, ha="right", fontsize=8)
    ax.set_title(f"{method.upper()} · {metric}: group − {result.reference_group}")
    fig.colorbar(im, ax=ax, shrink=.8, label="Effect")
    return fig


def render_figures(result: AnalysisResult, output_dir: Path, *,
                   inference: InferenceResult | None = None,
                   formats=("png", "pdf")) -> dict[str, list[Path]]:
    """Render every figure supported by the supplied data and metadata."""
    output_dir = Path(output_dir)
    outputs: dict[str, list[Path]] = {}
    def emit(name: str, fig: plt.Figure) -> None:
        outputs[name] = save_figure(fig, output_dir/name, formats=formats)
        plt.close(fig)
    emit("global_atoms", plot_global_atoms(result))
    emit("node_gradients", plot_node_gradients(result))
    emit("paired_mean_edge", plot_atom_pairs(result, inference=inference))
    for metric in ("global_efficiency", "modularity"):
        emit("paired_"+metric, plot_atom_pairs(result, measure=metric, inference=inference))
    if len(result.similarity):
        for reference in result.similarity.reference.unique():
            emit("similarity_"+_safe_name(reference),
                 plot_atom_pairs(result, reference=reference, inference=inference))
    if result.node_systems is not None:
        emit("system_gradients", plot_system_gradients(result))
    if inference is not None and len(result.group_order) > 1:
        for method in result.methods:
            for metric in ("rank_gradient", "balance", "sts", "rtr"):
                emit(f"regional_{_safe_name(method)}_{metric}",
                     plot_regional_effects(result, inference, method=method, metric=metric))
    return outputs
