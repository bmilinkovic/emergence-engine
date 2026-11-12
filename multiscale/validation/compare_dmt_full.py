"""Compare raw-source Python optimization checkpoints with archived MATLAB output.

Writes subject-level detail only to the caller-selected local output directory.
No participant-linked results are part of the package distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from multiscale.model import itransform_subspace


BANDS = ("delta", "theta", "alpha", "beta", "gamma")


def compare(outputs: Path, reference: Path):
    rows = []
    for condition in ("rest", "dmt"):
        for mat_file in sorted((reference / condition).glob("*_ssdi_spectral_domain_canonical.mat")):
            subject = mat_file.name.removesuffix("_ssdi_spectral_domain_canonical.mat")
            canonical = loadmat(mat_file, simplify_cells=True)["canonical"]
            with np.load(outputs / "models" / condition / f"{subject}.npz") as model:
                covariance = model["v"]
            for m in range(1, 5):
                for band in BANDS:
                    checkpoint = outputs / "blocks" / condition / subject / f"{band}_m{m}.npz"
                    if not checkpoint.exists():
                        continue
                    with np.load(checkpoint) as data:
                        py_pre = data["pre_values"]
                        py_final = data["final_values"]
                        py_clusters = len(data["representatives"])
                        py_best_basis = itransform_subspace(data["final_bases"][:, :, 0],
                                                             covariance)
                    matlab_pre = np.ravel(canonical[f"{band}_doptp_m{m}"])
                    matlab_final = np.ravel(canonical[f"{band}_dopto_m{m}"])
                    archived_basis = np.asarray(canonical[f"{band}_Lopto_m{m}"])
                    matlab_best_basis = (archived_basis[:, 0, None] if m == 1
                                         else archived_basis[:, :, 0])
                    projector_error = float(np.linalg.norm(
                        py_best_basis @ py_best_basis.T -
                        matlab_best_basis @ matlab_best_basis.T))
                    if len(py_pre) != len(matlab_pre):
                        raise ValueError(f"pre restart count mismatch: {condition}/{subject}/{band}/m{m}")
                    rows.append(dict(condition=condition, subject=subject,
                                     dimension=m, band=band,
                                     pre_max_abs=float(np.max(np.abs(py_pre-matlab_pre))),
                                     pre_best_abs=float(abs(py_pre[0]-matlab_pre[0])),
                                     python_final_count=len(py_final),
                                     matlab_final_count=len(matlab_final),
                                     python_cluster_count=py_clusters,
                                     final_best_abs=float(abs(py_final[0]-matlab_final[0])),
                                     best_projector_error=projector_error,
                                     final_max_abs=(float(np.max(np.abs(py_final-matlab_final)))
                                                    if len(py_final) == len(matlab_final) else None)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--detail", type=Path, help="local subject-level CSV")
    args = parser.parse_args()
    rows = compare(args.outputs, args.reference)
    if args.detail and rows:
        args.detail.parent.mkdir(parents=True, exist_ok=True)
        with args.detail.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(blocks_compared=len(rows), blocks_expected=760,
                   max_pre_abs=max((r["pre_max_abs"] for r in rows), default=None),
                   max_pre_best_abs=max((r["pre_best_abs"] for r in rows), default=None),
                   max_final_best_abs=max((r["final_best_abs"] for r in rows), default=None),
                   max_best_projector_error=max((r["best_projector_error"] for r in rows),
                                                default=None),
                   identical_final_count=sum(r["python_final_count"] == r["matlab_final_count"]
                                             for r in rows),
                   different_final_count=sum(r["python_final_count"] != r["matlab_final_count"]
                                             for r in rows),
                   max_final_abs_same_count=max((r["final_max_abs"] for r in rows
                                                 if r["final_max_abs"] is not None), default=None))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
