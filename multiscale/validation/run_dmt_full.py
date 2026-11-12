"""Resumable DMT raw-source to SSDI validation run.

This is a study-specific validation command, separate from the general
``multiscale`` package. Run ``--help`` for input paths and progress options.
No source or participant data are copied into the distribution.
"""

from __future__ import annotations

import os

# Set these before NumPy/SciPy are imported, including in spawned workers.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.io import loadmat

from multiscale import (fit_var_auto, prepare_var, opt_gd_ddx_mruns,
                        canonical_opt_dds_mruns)
from multiscale.model import itransform_subspace
from multiscale.optimise import cluster_representatives


BANDS = (("delta", 1, 4), ("theta", 5, 8), ("alpha", 9, 12),
         ("beta", 13, 30), ("gamma", 31, 44))
_CACHE = {}


def _raw_path(source_root, condition, subject):
    return source_root / condition / "data" / f"{subject}_macro_time_series.npy"


def _model_path(out, condition, subject):
    return out / "models" / condition / f"{subject}.npz"


def _block_path(out, condition, subject, band, m):
    return out / "blocks" / condition / subject / f"{band}_m{m}.npz"


def _fit_one(args):
    source_root, reference_models, out, condition, subject, regions, pmax, criterion = args
    started = time.monotonic()
    x = np.load(_raw_path(source_root, condition, subject))
    if x.ndim != 3 or x.shape[0] != regions:
        raise ValueError(f"unexpected source shape for {condition}/{subject}: {x.shape}")
    # fit_GC_subject_measures.m: pooled mean and sample standard deviation.
    mean = np.nanmean(x, axis=(1, 2), keepdims=True)
    scale = np.nanstd(x, axis=(1, 2), ddof=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale < np.finfo(float).eps)] = 1
    x = (x-mean)/scale
    selected = fit_var_auto(x, pmax, criterion=criterion, regmode="LWR")
    reference_path = reference_models / condition / f"{subject}_subject_measures.mat"
    with h5py.File(reference_path) as file:
        model = file["subject_measure/model"]
        reference_a = np.asarray(model["A"]).transpose(2, 1, 0)
        reference_v = np.asarray(model["V"]).T
        reference_order = int(np.asarray(model["morder"]).squeeze())
        meta = file["subject_measure/meta"]
        reference_ntime = int(np.asarray(meta["ntime"]).squeeze())
        reference_nepochs = int(np.asarray(meta["nep"]).squeeze())
    a, v = selected.fit.coefficients, selected.fit.covariance
    if a.shape != reference_a.shape:
        raise ValueError(f"model shape differs for {condition}/{subject}: {a.shape}, {reference_a.shape}")
    error_a = float(np.max(np.abs(a-reference_a)))
    error_v = float(np.max(np.abs(v-reference_v)))
    if (selected.order != reference_order or x.shape[1] != reference_ntime or
            x.shape[2] != reference_nepochs or error_a > 1e-8 or error_v > 1e-8):
        raise ValueError(f"raw-to-model parity failed for {condition}/{subject}: "
                         f"orders {selected.order}/{reference_order}, "
                         f"shapes {x.shape}, reference ({reference_ntime},{reference_nepochs}), "
                         f"errors {error_a:.3g}/{error_v:.3g}")
    target = _model_path(out, condition, subject)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".partial.npz")
    np.savez_compressed(temp, a=a, v=v, order=selected.order,
                        aic=selected.candidates.aic, bic=selected.candidates.bic,
                        hqc=selected.candidates.hqc, lrt=selected.candidates.lrt,
                        source_shape=np.array(x.shape))
    os.replace(temp, target)
    return dict(condition=condition, subject=subject, order=selected.order,
                error_a=error_a, error_v=error_v, seconds=time.monotonic()-started)


def _prepared(model_file, fs, fres):
    key = (str(model_file), fs, fres)
    if key not in _CACHE:
        with np.load(model_file) as data:
            a, v = data["a"].copy(), data["v"].copy()
        _, _, _, cak, h, freqs = prepare_var(a, v, fs, fres)
        _CACHE[key] = v, cak, h, freqs
    return _CACHE[key]


def _optimize_one(args):
    out, condition, subject, band, low, high, m, starts, pre_iters, final_iters, fs, fres = args
    start_time = time.monotonic()
    v, cak, h, freqs = _prepared(_model_path(out, condition, subject), fs, fres)
    indices = np.flatnonzero((freqs >= low) & (freqs <= high))
    pre = opt_gd_ddx_mruns(cak, starts, niters=pre_iters, gdes=2, sig0=1,
                           gdls=2, tol=1e-10, history=False)
    original = itransform_subspace(pre.bases, v)
    representatives, sizes = cluster_representatives(original, 1e-10)
    final = canonical_opt_dds_mruns(h, freqs, indices, pre.bases[:, :, representatives],
                                     niters=final_iters, gdes=2, sig0=0.1, gdls=2,
                                     tol=1e-10, history=False)
    target = _block_path(out, condition, subject, band, m)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".partial.npz")
    np.savez_compressed(temp, pre_values=pre.values, pre_bases=pre.bases,
                        pre_converged=pre.converged, pre_iterations=pre.iterations,
                        pre_step_sizes=pre.step_sizes,
                        representatives=representatives, cluster_sizes=sizes,
                        final_values=final.values, final_bases=final.bases,
                        final_converged=final.converged, final_iterations=final.iterations,
                        final_step_sizes=final.step_sizes)
    os.replace(temp, target)
    return dict(condition=condition, subject=subject, band=band, m=m,
                pre_runs=len(pre.values), final_runs=len(final.values),
                pre_iterations=int(np.sum(pre.iterations)),
                final_iterations=int(np.sum(final.iterations)),
                pre_best=float(pre.values[0]), final_best=float(final.values[0]),
                seconds=time.monotonic()-start_time)


def _log(out, message):
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with (out / "progress.log").open("a") as file:
        file.write(line+"\n")


def _progress(out, label, done, total, started, detail):
    elapsed = time.monotonic()-started
    remaining = elapsed*max(total-done, 0)/max(done, 1)
    filled = int(24*done/total)
    bar = "#"*filled+"."*(24-filled)
    _log(out, f"{label} [{bar}] {done}/{total} ({100*done/total:.1f}%) "
              f"elapsed={elapsed/3600:.2f}h ETA={remaining/3600:.2f}h {detail}")
    state = dict(stage=label, completed=done, total=total,
                 percent=100*done/total, elapsed_seconds=elapsed,
                 eta_seconds=remaining, last=detail)
    (out / "status.json").write_text(json.dumps(state, indent=2)+"\n")


def _jobs(source_root):
    jobs = []
    for condition in ("rest", "dmt"):
        files = sorted((source_root / condition / "data").glob("*_macro_time_series.npy"))
        for file in files:
            jobs.append((condition, file.name.removesuffix("_macro_time_series.npy")))
    return jobs


def _fit_all(args, jobs):
    out = args.output
    todo = [(args.source_root, args.reference_models, out, condition, subject,
             args.regions, args.pmax, args.criterion)
            for condition, subject in jobs
            if not _model_path(out, condition, subject).exists()]
    started = time.monotonic()
    done = len(jobs)-len(todo)
    _progress(out, "FIT", done, len(jobs), started, "starting/resuming")
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_fit_one, item) for item in todo]
        for future in as_completed(futures):
            result = future.result()
            done += 1
            _progress(out, "FIT", done, len(jobs), started,
                      f"{result['condition']}/{result['subject']} p={result['order']} "
                      f"Aerr={result['error_a']:.2e} Verr={result['error_v']:.2e}")


def _optimize_all(args, jobs):
    if not args.starts_file:
        raise ValueError("--starts-file is required for the MATLAB-matched run")
    starts = loadmat(args.starts_file, simplify_cells=True)["starts"]
    tasks = []
    for condition, subject in jobs:
        for m in range(args.m_min, args.m_max+1):
            for band_number, (band, low, high) in enumerate(BANDS, start=1):
                key = f"{condition}_{subject}_m{m}_b{band_number}"
                l0 = np.asarray(starts[key])
                # scipy's simplify_cells squeezes MATLAB singleton dimensions.
                if m == 1 and l0.ndim == 2:
                    l0 = l0[:, None, :]
                if l0.shape != (args.regions, m, 100):
                    raise ValueError(f"unexpected initial bases for {key}: {l0.shape}")
                tasks.append((args.output, condition, subject, band, low, high, m,
                              l0, args.pre_iters, args.final_iters, args.fs, args.fres))
    if args.limit_blocks:
        tasks = tasks[:args.limit_blocks]
    for item in tasks:
        model_path = _model_path(args.output, item[1], item[2])
        if not model_path.exists():
            raise FileNotFoundError(model_path)
    started = time.monotonic()
    todo = [item for item in tasks if not _block_path(args.output, item[1], item[2], item[3], item[6]).exists()]
    done = len(tasks)-len(todo)
    _progress(args.output, "SSDI", done, len(tasks), started, "starting/resuming")
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_optimize_one, item) for item in todo]
        for future in as_completed(futures):
            result = future.result()
            done += 1
            _progress(args.output, "SSDI", done, len(tasks), started,
                      f"{result['condition']}/{result['subject']} {result['band']} "
                      f"m={result['m']} final_runs={result['final_runs']} "
                      f"best={result['final_best']:.6g} block={result['seconds']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="archived source-reconstructed root with rest/dmt/data/*.npy")
    parser.add_argument("--reference-models", type=Path, required=True,
                        help="archived MATLAB subject_measures root")
    parser.add_argument("--starts-file", type=Path,
                        help="MATLAB-generated starts struct for matched trajectory validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--regions", type=int, default=5)
    parser.add_argument("--m-min", type=int, default=1)
    parser.add_argument("--m-max", type=int, default=4)
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--fres", type=int, default=256)
    parser.add_argument("--pmax", type=int, default=30)
    parser.add_argument("--criterion", choices=("aic", "bic", "hqc", "lrt"),
                        default="lrt")
    parser.add_argument("--stage", choices=("fit", "optimize", "both"), default="both")
    parser.add_argument("--pre-iters", type=int, default=10000)
    parser.add_argument("--final-iters", type=int, default=10000)
    parser.add_argument("--limit-blocks", type=int, help="smoke-test a prefix of blocks")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(args.source_root)
    if len(jobs) != 38:
        raise ValueError(f"expected 38 DMT source files, found {len(jobs)}")
    if not 1 <= args.m_min <= args.m_max < args.regions:
        raise ValueError("macro dimensions must satisfy 1 <= min <= max < regions")
    _log(args.output, f"workers={args.workers} subjects/conditions={len(jobs)} "
                      f"regions={args.regions} m={args.m_min}:{args.m_max} "
                      f"fs={args.fs} fres={args.fres} pmax={args.pmax} "
                      f"criterion={args.criterion} pre_iters={args.pre_iters} "
                      f"final_iters={args.final_iters}")
    if args.stage in ("fit", "both"):
        _fit_all(args, jobs)
    if args.stage in ("optimize", "both"):
        _optimize_all(args, jobs)
    _log(args.output, "run complete")


if __name__ == "__main__":
    main()
