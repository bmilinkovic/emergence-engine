"""Exact pairwise panel functions copied from the original DoC analysis."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
COHORT_ORDER = ["control", "emcs", "mcs", "uws", "coma"]
COHORT_DISPLAY = {"control":"CNTL","emcs":"EMCS","mcs":"MCS","uws":"UWS","coma":"COMA"}
METHOD_ORDER = ["mmi","ccs"]
METHOD_LABELS = {"mmi":"MMI","ccs":"CCS"}
ATOM_COLORS = {"rtr":"#2E5EAA","sts":"#D43D2A"}

def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12.0,
            "axes.titlesize": 21.0,
            "axes.labelsize": 18.0,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_transparent.png", dpi=320, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _star(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def _safe_sem(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def _add_bracket(ax: plt.Axes, x0: float, x1: float, y: float, text: str, color: str = "#1F2430") -> None:
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = 0.025 * yr
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=1.2, clip_on=False)
    ax.text((x0 + x1) / 2.0, y + h + 0.008 * yr, text, ha="center", va="bottom", fontsize=14, fontweight="bold", color=color)


def _plot_similarity_paired(
    subject_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    rtr_col: str,
    sts_col: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
) -> None:
    _set_style()
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.6), sharex=True, squeeze=False)
    axes = axes[:, 0]
    offsets = {"rtr": -0.18, "sts": 0.18}
    rng = np.random.default_rng(123)
    for row_idx, method in enumerate(METHOD_ORDER):
        ax = axes[row_idx]
        sub_method = subject_df.loc[subject_df["method"] == method].copy()
        yvals = []
        for cidx, cohort in enumerate(COHORT_ORDER):
            sub = sub_method.loc[sub_method["cohort"] == cohort].copy()
            for atom, col in [("rtr", rtr_col), ("sts", sts_col)]:
                vals = sub[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                x = cidx + offsets[atom]
                jitter = rng.normal(0.0, 0.05, size=vals.size)
                ax.scatter(np.full(vals.size, x) + jitter, vals, s=42, color=ATOM_COLORS[atom], alpha=0.28, edgecolors="none", zorder=2)
                mean = float(np.nanmean(vals)) if vals.size else np.nan
                sem = _safe_sem(vals)
                ax.errorbar(x, mean, yerr=sem, color="#1E1E1E", capsize=3, lw=1.2, zorder=4)
                ax.scatter(x, mean, s=64, color=ATOM_COLORS[atom], edgecolor="black", linewidth=0.4, zorder=5)
                yvals.extend(list(vals))

            hit = stats_df.loc[(stats_df["method"] == method) & (stats_df["cohort"] == cohort)]
            if not hit.empty:
                stars = _star(float(hit["perm_q_fdr"].iloc[0]))
                if stars:
                    max_y = np.nanmax(
                        np.concatenate(
                            [
                                sub[rtr_col].to_numpy(dtype=float),
                                sub[sts_col].to_numpy(dtype=float),
                            ]
                        )
                    )
                    _add_bracket(ax, cidx + offsets["rtr"], cidx + offsets["sts"], max_y + 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9), stars)

        ax.axhline(0.0, color="#BEB7A7", lw=0.9, ls="--", zorder=0)
        ax.set_ylabel(ylabel)
        ax.text(-0.12, 0.5, METHOD_LABELS[method], rotation=90, va="center", ha="center", transform=ax.transAxes, fontsize=22, color="#1F2430")
        pad = 0.18 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9)
        ax.set_ylim(np.nanmin(yvals) - 0.08 * pad, np.nanmax(yvals) + pad)
    axes[-1].set_xticks(range(len(COHORT_ORDER)))
    axes[-1].set_xticklabels([COHORT_DISPLAY[c] for c in COHORT_ORDER])
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATOM_COLORS["rtr"], markersize=10, label="Redundancy (RTR)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATOM_COLORS["sts"], markersize=10, label="Synergy (STS)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.968), ncol=2, fontsize=16)
    fig.suptitle(title, fontsize=24, y=0.992)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.11, hspace=0.16)
    _save_figure(fig, out_dir, stem)


def _plot_graph_paired(
    graph_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str,
) -> None:
    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.8), squeeze=False)
    offsets = {"rtr": -0.18, "sts": 0.18}
    rng = np.random.default_rng(321)
    metric_titles = {"global_efficiency": "Global efficiency", "modularity": "Modularity"}
    for row_idx, method in enumerate(METHOD_ORDER):
        sub_method = graph_df.loc[graph_df["method"] == method].copy()
        for col_idx, metric in enumerate(["global_efficiency", "modularity"]):
            ax = axes[row_idx, col_idx]
            yvals = []
            for cidx, cohort in enumerate(COHORT_ORDER):
                sub = sub_method.loc[sub_method["cohort"] == cohort].copy()
                for atom in ["rtr", "sts"]:
                    vals = sub.loc[sub["atom"] == atom, metric].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    x = cidx + offsets[atom]
                    jitter = rng.normal(0.0, 0.05, size=vals.size)
                    ax.scatter(np.full(vals.size, x) + jitter, vals, s=40, color=ATOM_COLORS[atom], alpha=0.28, edgecolors="none", zorder=2)
                    mean = float(np.nanmean(vals)) if vals.size else np.nan
                    sem = _safe_sem(vals)
                    ax.errorbar(x, mean, yerr=sem, color="#1E1E1E", capsize=3, lw=1.2, zorder=4)
                    ax.scatter(x, mean, s=64, color=ATOM_COLORS[atom], edgecolor="black", linewidth=0.4, zorder=5)
                    yvals.extend(list(vals))
                hit = stats_df.loc[(stats_df["method"] == method) & (stats_df["cohort"] == cohort) & (stats_df["measure"] == metric)]
                if not hit.empty:
                    stars = _star(float(hit["perm_q_fdr"].iloc[0]))
                    if stars:
                        max_y = np.nanmax(np.concatenate([sub.loc[sub["atom"] == "rtr", metric].to_numpy(dtype=float), sub.loc[sub["atom"] == "sts", metric].to_numpy(dtype=float)]))
                        _add_bracket(ax, cidx + offsets["rtr"], cidx + offsets["sts"], max_y + 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9), stars)
            ax.set_xticks(range(len(COHORT_ORDER)))
            if row_idx == 1:
                ax.set_xticklabels([COHORT_DISPLAY[c] for c in COHORT_ORDER])
            else:
                ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(metric_titles[metric])
            if col_idx == 0:
                ax.set_ylabel("Value")
                ax.text(-0.20, 0.5, METHOD_LABELS[method], rotation=90, va="center", ha="center", transform=ax.transAxes, fontsize=22, color="#1F2430")
            ax.grid(axis="y", color="#E8E1D3", linewidth=0.7, alpha=0.7)
            ax.set_axisbelow(True)
            pad = 0.18 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-9)
            ax.set_ylim(np.nanmin(yvals) - 0.08 * pad, np.nanmax(yvals) + pad)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATOM_COLORS["rtr"], markersize=10, label="Redundancy (RTR)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATOM_COLORS["sts"], markersize=10, label="Synergy (STS)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, fontsize=16)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.11, wspace=0.18, hspace=0.18)
    _save_figure(fig, out_dir, stem)

# Public names for the retained upstream plotting implementations.
plot_similarity_paired = _plot_similarity_paired
plot_graph_paired = _plot_graph_paired
