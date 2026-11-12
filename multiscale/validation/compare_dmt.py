"""Read-only comparison with archived DMT canonical MATLAB results.

Usage: python validation/compare_dmt.py RESULTS_ROOT MODEL_ROOT [--temporal]
Paths are supplied explicitly; no participant data is bundled with the wheel.
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from multiscale import prepare_var, canonical_band_value, iss2dd
from multiscale.model import orthonormalise


def compare_file(path, model_root, temporal=False, first_run=False,
                 temporal_one_per_file=False):
    saved = loadmat(path, simplify_cells=True)["canonical"]
    meta = saved["meta"]
    subject, condition = str(meta["participant_id"]), str(meta["condition"])
    model_path = model_root / condition / f"{subject}_subject_measures.mat"
    with h5py.File(model_path) as model:
        a = np.asarray(model["subject_measure/model/A"]).transpose(2, 1, 0)
        v = np.asarray(model["subject_measure/model/V"]).T
    ass, c, k, _, h, freqs = prepare_var(a, v, int(meta["fs"]), int(meta["fres"]))
    root = np.linalg.cholesky(v)
    rows = []
    temporal_done = False
    for band in meta["band_info"]:
        name = str(band["name"])
        idx = np.flatnonzero((freqs >= band["lo"]) & (freqs <= band["hi"]))
        if not len(idx):
            continue
        for m in range(2, a.shape[0]):
            key = f"{name}_Lopto_m{m}"
            if key not in saved:
                continue
            bases = np.asarray(saved[key])
            if bases.ndim == 2:
                bases = bases[:, :, None]
            reference = np.atleast_1d(saved[f"{name}_canonical_band_dd_m{m}"])
            for run in range(1 if first_run else bases.shape[2]):
                l = orthonormalise(root.T @ bases[:, :, run])
                computed = canonical_band_value(l, h, freqs, idx)
                ok = np.isfinite(computed)
                row = dict(subject=subject, condition=condition, band=name, m=m,
                           run=run+1, spectral_reference=reference[run],
                           spectral_computed=computed, spectral_abs_error=abs(computed-reference[run]),
                           valid=ok)
                if temporal and run == 0 and (not temporal_one_per_file or not temporal_done):
                    tref = np.atleast_1d(saved[f"{name}_time_dd_dare_m{m}"])[run]
                    tvalue = iss2dd(l, ass, c, k)
                    row.update(temporal_reference=tref, temporal_computed=tvalue,
                               temporal_abs_error=abs(tvalue-tref))
                    temporal_done = True
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--temporal", action="store_true", help="check first run per band and dimension")
    parser.add_argument("--temporal-one-per-file", action="store_true",
                        help="check one temporal result per subject/condition file")
    parser.add_argument("--first-run", action="store_true", help="fast screening: first run only")
    parser.add_argument("--output", type=Path, help="CSV report path")
    args = parser.parse_args()
    files = sorted(args.results_root.glob("*/*_ssdi_spectral_domain_canonical.mat"))
    if not files:
        parser.error("no canonical MATLAB result files found")
    rows = []
    for path in files:
        try:
            batch = compare_file(path, args.model_root,
                                 temporal=args.temporal or args.temporal_one_per_file,
                                 first_run=args.first_run,
                                 temporal_one_per_file=args.temporal_one_per_file)
            rows.extend(batch)
            err = max((row["spectral_abs_error"] for row in batch), default=np.nan)
            print(f"{path.parent.name}/{path.name}: {len(batch)} comparisons, max spectral error {err:.3g}", flush=True)
        except Exception as exc:
            print(f"FAILED {path}: {exc}", file=sys.stderr, flush=True)
            raise
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
    print(f"TOTAL {len(rows)} comparisons; maximum spectral absolute error "
          f"{max(r['spectral_abs_error'] for r in rows):.3g}")
    if args.temporal or args.temporal_one_per_file:
        print("maximum temporal absolute error "
              f"{max(r['temporal_abs_error'] for r in rows if 'temporal_abs_error' in r):.3g}")
    if max(r["spectral_abs_error"] for r in rows) > 1e-8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
