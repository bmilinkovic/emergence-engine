"""Render and audit archived DMT figures with reusable multiscale modules.

Study data and subject-level tables stay in the caller-selected local output.
The publication package contains only general analysis/statistics/plotting code.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multiscale.analysis import (load_matlab_full, load_python_full,
                                 load_timewindow_tables)
from multiscale.statistics import (grand_average_dd, paired_tests,
                                   pooled_sf_stats, window_state_table,
                                   state_pairwise_tests)
from multiscale.visualisation import (
    plot_dd_grand_average, plot_dd_violin_by_macro, plot_sf_index,
    plot_similarity_matrices, plot_timewindow_dd, plot_timewindow_sf)
from multiscale.visualisation import plot_state_trajectory, plot_pairwise_heatmap


def _parse_named(values):
    out = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep:
            raise ValueError("inputs must use NAME=PATH")
        out[name] = Path(path)
    return out


def _full(label, dd, sf, matrices, output):
    output.mkdir(parents=True, exist_ok=True)
    if len(dd) == 0 or len(sf) == 0:
        raise ValueError(f"empty full-recording results: {label}")
    dd_stats = paired_tests(dd, "dd_min", "rest", "dmt", ["m", "band"], ["m"])
    sf_stats = paired_tests(sf, "sf_index", "rest", "dmt",
                            ["band", "m"], ["band"])
    grand = grand_average_dd(dd, "rest", "dmt", paired=False)
    pooled = pooled_sf_stats(sf, "rest", "dmt")
    for name, table in (("dd_subject_long", dd), ("sf_subject_long", sf),
                        ("dd_paired_stats", dd_stats),
                        ("dd_grand_stats", grand),
                        ("sf_paired_stats", sf_stats),
                        ("sf_pooled_stats", pooled)):
        table.to_csv(output / f"{name}.csv", index=False)
    figures = output / "figures"
    plot_dd_violin_by_macro(dd, dd_stats, figures / "canonical_dd_min_violin_by_macro")
    plot_dd_grand_average(dd, grand, figures / "canonical_dd_min_grand_average")
    plot_sf_index(sf, sf_stats, figures / "sf_index_paired_main")
    plot_sf_index(sf, pooled, figures / "sf_index_pooled_main")
    plot_similarity_matrices(matrices, figures / "sf_similarity_matrices_supplementary")
    return dict(label=label, blocks=len(dd), sf_blocks=len(sf),
                dd_significant=int(dd_stats.significant.sum()),
                sf_significant=int(sf_stats.significant.sum()),
                figures=5)


def _timewindow(label, root, output):
    output.mkdir(parents=True, exist_ok=True)
    dd, sf = load_timewindow_tables(root)
    dd_stats = paired_tests(dd, "dd_min", "rest", "dmt",
                             ["window", "m", "band"], ["window", "m"])
    sf_stats = paired_tests(sf, "sf_index", "rest", "dmt",
                             ["window", "band", "m"], ["window", "band"])
    grand = grand_average_dd(dd, "rest", "dmt", paired=True)
    for name, table in (("dd_paired_stats", dd_stats),
                        ("sf_paired_stats", sf_stats),
                        ("dd_grand_stats", grand)):
        table.to_csv(output / f"{name}.csv", index=False)
    figures = output / "figures"
    plot_timewindow_dd(dd, figures / "grand_average_DD_composite_timewindows")
    plot_timewindow_sf(sf, figures / "SF_index_4x5_timewindow_band_composite")
    states = ["Rest", *dict.fromkeys(dd.window)]
    for name, table, value in (("dd", dd, "dd_min"),
                               ("sf", sf, "sf_index")):
        state = window_state_table(table, value)
        detail, pair_grand = state_pairwise_tests(
            state, value, states,
            detail_correction_groups=("band",) if name == "sf" else ("m",))
        state.to_csv(output / f"{name}_state_long.csv", index=False)
        detail.to_csv(output / f"{name}_state_pairwise_stats.csv", index=False)
        pair_grand.to_csv(output / f"{name}_state_pairwise_grand_stats.csv", index=False)
        plot_state_trajectory(state, value, states,
                              figures / f"{name}_state_trajectory_by_band")
        plot_pairwise_heatmap(pair_grand, states,
                              figures / f"{name}_pairwise_state_heatmap_by_band")
    return dict(label=label, dd_rows=len(dd), sf_rows=len(sf),
                dd_significant=int(dd_stats.significant.sum()),
                sf_significant=int(sf_stats.significant.sum()),
                figures=6)


def _difference(left, right, keys, columns):
    joined = left.merge(right, on=keys, suffixes=("_python", "_archived"))
    result = dict(rows=len(joined), expected=len(right))
    for column in columns:
        a = joined[f"{column}_python"].to_numpy(float)
        b = joined[f"{column}_archived"].to_numpy(float)
        result[f"max_abs_{column}"] = float(np.nanmax(abs(a-b)))
        result[f"median_abs_{column}"] = float(np.nanmedian(abs(a-b)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-full", action="append", default=[],
                        help="NAME=ROOT, repeatable")
    parser.add_argument("--timewindow", action="append", default=[],
                        help="NAME=ROOT, repeatable")
    parser.add_argument("--python-full", type=Path,
                        help="Python full-optimization checkpoint root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_file = args.output / "parity_summary.json"
    report = (json.loads(report_file.read_text()) if report_file.exists()
              else {"full": {}, "timewindows": {}, "parity": {}})
    full_roots = _parse_named(args.matlab_full)
    for label, root in full_roots.items():
        print(f"Loading MATLAB full {label}", flush=True)
        dd, sf, matrices = load_matlab_full(root)
        report["full"][label] = _full(label, dd, sf, matrices, args.output / label)
        archived_dd = pd.read_csv(root / "summaries" / "canonical_dd_min_subject_long.csv")
        archived_sf = pd.read_csv(root / "summaries" / "similarity_sf_lopto" /
                                  "sf_index_subject_long.csv")
        report["parity"][f"{label}_archive_rebuild_dd"] = _difference(
            dd, archived_dd, ["subject", "condition", "band", "m"], ["dd_min"])
        report["parity"][f"{label}_archive_rebuild_sf"] = _difference(
            sf, archived_sf, ["subject", "condition", "band", "m"], ["sf_index"])
    if args.python_full:
        print("Loading Python full checkpoints and computing similarities", flush=True)
        dd, sf, matrices = load_python_full(args.python_full)
        report["full"]["python_momax30"] = _full(
            "python_momax30", dd, sf, matrices, args.output / "python_momax30")
        if "momax30" in full_roots:
            matlab_dd = pd.read_csv(args.output / "momax30" / "dd_subject_long.csv")
            matlab_sf = pd.read_csv(args.output / "momax30" / "sf_subject_long.csv")
            report["parity"]["python_vs_matlab_dd"] = _difference(
                dd, matlab_dd, ["subject", "condition", "band", "m"], ["dd_min"])
            report["parity"]["python_vs_matlab_sf"] = _difference(
                sf, matlab_sf, ["subject", "condition", "band", "m"], ["sf_index"])
    for label, root in _parse_named(args.timewindow).items():
        print(f"Loading archived time windows {label}", flush=True)
        report["timewindows"][label] = _timewindow(label, root,
                                                    args.output / label)
    report_file.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
