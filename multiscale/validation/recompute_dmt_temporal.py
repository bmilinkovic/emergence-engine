"""Recompute archived DK5 temporal DI using covariance-balanced QZ DARE.

This is a study-specific repair of the NaN temporal fields in the archived
canonical files. It saves independent local checkpoints; it never edits the
MATLAB archive or includes participant data in the package distribution.
"""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat

from multiscale import prepare_var, mdare
from multiscale.model import orthonormalise
from multiscale.riccati import covariance_balance_iss


BANDS = ("delta", "theta", "alpha", "beta", "gamma")


def _one(task):
    mat_file, models, output = task
    start = time.monotonic()
    condition = mat_file.parent.name
    subject = mat_file.name.removesuffix("_ssdi_spectral_domain_canonical.mat")
    with np.load(models / condition / f"{subject}.npz") as model:
        a, v = model["a"], model["v"]
    archived = loadmat(mat_file, simplify_cells=True)["canonical"]
    meta = archived["meta"]
    ass, c, k, _, _, _ = prepare_var(a, v, int(meta["fs"]), int(meta["fres"]))
    ass, c, k = covariance_balance_iss(ass, c, k)
    root = np.linalg.cholesky(v)
    q = k @ k.T
    results = {}
    finite = total = 0
    largest_residual = 0.0
    for m in range(1, a.shape[0]):
        for band in BANDS:
            key = f"{band}_m{m}"
            bases = np.asarray(archived[f"{band}_Lopto_m{m}"])
            if m == 1 and bases.ndim == 2:
                bases = bases[:, None, :]
            elif bases.ndim == 2:
                bases = bases[:, :, None]
            dd = np.full(bases.shape[2], np.nan)
            residuals = np.full(bases.shape[2], np.nan)
            for run in range(bases.shape[2]):
                l = orthonormalise(root.T @ bases[:, :, run])
                try:
                    _, innovation, residual, _ = mdare(
                        ass, l.T @ c, q, np.eye(m), k @ l)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                residuals[run] = residual
                if innovation is not None and 0 <= residual <= 1e-8:
                    sign, logdet = np.linalg.slogdet(innovation)
                    if sign > 0:
                        dd[run] = logdet
            results[key] = dd
            results[f"residual_{key}"] = residuals
            total += len(dd)
            finite += int(np.isfinite(dd).sum())
            valid_residuals = residuals[np.isfinite(residuals) & (residuals >= 0)]
            if valid_residuals.size:
                largest_residual = max(largest_residual, float(valid_residuals.max()))
    target = output / condition / f"{subject}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".partial.npz")
    np.savez_compressed(temp, **results)
    os.replace(temp, target)
    return dict(condition=condition, subject=subject, finite=finite, total=total,
                max_residual=largest_residual, seconds=time.monotonic()-start)


def _log(output, done, total, result):
    elapsed = result["elapsed"]
    eta = elapsed*(total-done)/max(done, 1)
    line = (f"{datetime.now().astimezone().isoformat(timespec='seconds')} "
            f"DARE {done}/{total} ({100*done/total:.1f}%) ETA={eta/60:.1f}min "
            f"{result['condition']}/{result['subject']} "
            f"finite={result['finite']}/{result['total']} "
            f"max_residual={result['max_residual']:.3g}")
    print(line, flush=True)
    with (output / "temporal_progress.log").open("a") as file:
        file.write(line+"\n")
    (output / "temporal_status.json").write_text(json.dumps(
        dict(completed=done, total=total, elapsed_seconds=elapsed,
             eta_seconds=eta, last=result), indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="MATLAB DK5 canonical root")
    parser.add_argument("models", type=Path, help="raw-fitted Python model checkpoints")
    parser.add_argument("output", type=Path, help="local repaired temporal root")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.reference.glob("*/*_ssdi_spectral_domain_canonical.mat"))
    if not files:
        parser.error("no canonical files found")
    tasks = [(file, args.models, args.output) for file in files
             if not (args.output / file.parent.name /
                     f"{file.name.removesuffix('_ssdi_spectral_domain_canonical.mat')}.npz").exists()]
    started = time.monotonic()
    done = len(files)-len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            done += 1
            result["elapsed"] = time.monotonic()-started
            _log(args.output, done, len(files), result)
    print(f"repaired {done}/{len(files)} files", flush=True)


if __name__ == "__main__":
    main()
