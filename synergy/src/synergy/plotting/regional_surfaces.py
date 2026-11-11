"""Regional surface figure functions copied from the original DoC analysis."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from . import atlas_surfaces as style
from ..analysis.regional import regional_mean,gradient,subject_values
from ..statistics.regional import regional_tests,global_tests,coma_global_sensitivity
COHORTS=("control","emcs","mcs","uws","coma")
METHODS=("mmi","ccs")

def aal_surface_names(names: list[str]) -> np.ndarray:
    labels = pd.read_csv(style.ATLAS.with_suffix(".txt"), sep="\t", header=None,
                         names=["abbreviation", "roi_label", "code"])
    lookup = dict(zip(labels.roi_label, labels.code))
    if any(name not in lookup for name in names):
        raise ValueError("One or more DoC ROI labels are absent from AAL116 atlas")
    return np.asarray([f"aal116.cluster{int(lookup[name])}" for name in names])


def plot_global(global_df: pd.DataFrame, tests: pd.DataFrame, method: str,
                pipeline: str, folder: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8))
    fig.subplots_adjust(left=.08, right=.99, top=.83, bottom=.19, wspace=.25)
    palette = {"sts": style.ORANGE, "rtr": style.GREEN, "balance": "#ed9220"}
    titles = {"sts": "Synergy", "rtr": "Redundancy",
              "balance": "Raw synergy − redundancy"}
    rng = np.random.default_rng(37)
    for ax, atom in zip(axes, ("sts", "rtr", "balance")):
        sub = global_df[(global_df.method == method) & (global_df.atom == atom)]
        arrays = [sub[sub.cohort == c].mean_edge_value.to_numpy(float) for c in COHORTS]
        violins = ax.violinplot(arrays, positions=np.arange(5), widths=.62,
                                showmeans=False, showmedians=False, showextrema=False)
        for body in violins["bodies"]:
            body.set_facecolor(palette[atom]); body.set_edgecolor(palette[atom])
            body.set_alpha(.18); body.set_linewidth(1.3)
        for i, values in enumerate(arrays):
            x = i + rng.uniform(-.085, .085, size=len(values))
            ax.scatter(x, values, s=24, zorder=4,
                       facecolors="white" if i == 0 else palette[atom],
                       edgecolors=palette[atom] if i == 0 else "white",
                       linewidths=1.0 if i == 0 else .4)
            median = float(np.median(values))
            ax.plot([i-.13, i+.13], [median, median], color=style.INK, lw=2.2, zorder=5)
        low = min(v.min() for v in arrays); high = max(v.max() for v in arrays)
        span = max(high-low, 1e-6)
        for i, cohort in enumerate(COHORTS[1:], 1):
            q = float(tests[(tests.method == method) & (tests.atom == atom) &
                            (tests.cohort == cohort)].q_fdr_12_per_method.iloc[0])
            stars = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
            if stars:
                ax.text(i, high+.035*span, stars, ha="center", va="bottom", fontsize=13,
                        color=style.INK, weight="bold")
        ax.set_title(titles[atom], fontsize=15, pad=12)
        ax.set_ylim(low-.08*span, high+.16*span)
        ax.set_xlim(-.45, 4.45)
        ax.set_xticks(range(5), ["Control", "EMCS", "MCS", "UWS", "Coma"])
        ax.tick_params(axis="both", labelsize=10, length=0, pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#819089")
        ax.grid(False); ax.patch.set_alpha(0)
        if atom == "sts": ax.set_ylabel("Mean valid edge value (nats)", fontsize=11)
    fig.suptitle(f"Global PhiID atoms by DoC cohort · {method.upper()} · {pipeline}",
                 fontsize=16, weight="bold", color=style.INK)
    fig.text(.5, .045, "Stars: patient versus control, unpaired permutation test; FDR within 12 tests per estimator",
             ha="center", va="center", fontsize=9, color=style.INK)
    for label in fig.findobj(match=Text):
        label.set_fontfamily("Helvetica"); label.set_fontweight("bold")
    style.save(fig, folder, "global_atoms_by_cohort")


def plot_doc(pipeline: str, meshes, input_dir: Path, output_dir: Path, mode: str = "all") -> None:
    out = Path(output_dir)
    source = Path(input_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads((source / "roi_labels.json").read_text())
    surface_names = aal_surface_names(names)
    nodal, global_df, matrices = subject_values(source, names)
    regional = regional_tests(nodal)
    global_stats = global_tests(global_df)
    nodal.to_csv(out / "subject_drug_style_regional_values.csv", index=False)
    global_df.to_csv(out / "subject_raw_atom_balance.csv", index=False)
    regional.to_csv(out / "patient_vs_control_regional_rank_balance_tests.csv", index=False)
    global_stats.to_csv(out / "patient_vs_control_global_atom_tests.csv", index=False)
    coma_global_sensitivity(global_df, pd.read_csv(source / "edge_validity_qc.csv")).to_csv(
        out / "coma_global_atom_sedation_and_invalid_roi_sensitivity.csv", index=False)
    for method in METHODS:
        folder = out / "figures" / "regional_balance" / method
        condition = []
        for cohort in COHORTS:
            with np.errstate(invalid="ignore"):
                sts = np.nanmean(np.stack(matrices[(method, cohort, "sts")]), axis=0)
                rtr = np.nanmean(np.stack(matrices[(method, cohort, "rtr")]), axis=0)
            condition.append((cohort.upper(), gradient(sts, rtr)))
        if mode == "all":
            style.render_surface_rows(
                "DoC", surface_names, condition, 90, "regional_rank_conditions",
                folder, meshes, title=f"Synergy − redundancy rank · {method.upper()} · {pipeline}",
                unit="rank", bar_ends=("Redundant", "Synergistic"))
        if mode in ("all", "contrast", "refresh", "components"):
          for metric, filename, title, cmap, ends in (
            ("rank_gradient", "regional_rank_change_vs_control",
             f"Change in synergy − redundancy rank: patient − control · {method.upper()} · {pipeline}",
             style.RANK_CMAP, ("Toward redundancy", "Toward synergy")),
            ("balance", "regional_raw_balance_change_vs_control",
             f"Raw Δ(STS − RTR): patient − control · {method.upper()} · {pipeline}",
             style.DRUG_CHANGE_CMAP, ("Lower in patient", "Higher in patient")),
            ("sts", "regional_raw_sts_change_vs_control",
             f"Raw ΔSTS: patient − control · {method.upper()} · {pipeline}",
             style.DRUG_CHANGE_CMAP, ("Lower in patient", "Higher in patient")),
            ("rtr", "regional_raw_rtr_change_vs_control",
             f"Raw ΔRTR: patient − control · {method.upper()} · {pipeline}",
             style.DRUG_CHANGE_CMAP, ("Lower in patient", "Higher in patient")),
          ):
              if mode == "components" and metric not in ("sts", "rtr"):
                  continue
              if mode not in ("all", "components") and metric in ("sts", "rtr"):
                  continue
              rows, sigs = [], []
              for cohort in COHORTS[1:]:
                  sub = regional[(regional.method == method) &
                                 (regional.metric == metric) & (regional.cohort == cohort)]
                  rows.append((cohort.upper(),
                               sub.patient_minus_control.to_numpy(float)))
                  sigs.append((sub.q_fdr_360_per_method_metric.to_numpy(float) < .05))
              vals = np.concatenate([x[1][np.isfinite(x[1])] for x in rows])
              if metric == "rank_gradient":
                  bound = max(25, float(np.ceil(np.percentile(np.abs(vals), 98)/5)*5))
              else:
                  raw = max(float(np.percentile(np.abs(vals), 98)), 1e-6)
                  step = 10**np.floor(np.log10(raw/10))
                  bound = float(np.ceil(raw/step)*step)
              style.render_surface_rows(
                  "DoC", surface_names, rows, bound, filename, folder, meshes,
                  title=title, unit="Δ rank" if metric == "rank_gradient" else "Δ nats",
                  bar_ends=ends, cmap=cmap, significant=np.stack(sigs))
        if mode in ("all", "global", "refresh"):
            plot_global(global_df, global_stats, method, pipeline, folder)
        print(f"DoC {pipeline} {method}: regional figures complete", flush=True)
