"""Turn optimized DD and subspaces into tidy, atlas-independent tables."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .model import itransform_subspace
from .optimise import _distance
from .statistics import sf_index


def similarity_matrix(bases):
    """SSDI-1 maximum-principal-angle Grassmann distance for all run pairs."""
    bases = np.asarray(bases, float)
    if bases.ndim != 3:
        raise ValueError("bases must have shape (regions, dimension, runs)")
    count = bases.shape[2]
    out = np.zeros((count, count))
    for i in range(count):
        for j in range(i+1, count):
            out[i, j] = out[j, i] = _distance(bases[:, :, i], bases[:, :, j])
    return out


def _summarize(condition, subject, band, m, values, similarity, threshold):
    values = np.ravel(np.asarray(values, float))
    n = len(values)
    if n < 1:
        raise ValueError("each block needs at least one DD value")
    dd = dict(subject=subject, condition=condition, band=band, m=m,
              dd_min=float(np.min(values)), dd_median=float(np.median(values)),
              dd_mean=float(np.mean(values)), n_runs=n)
    g = np.asarray(similarity, float)
    if g.shape != (n, n):
        raise ValueError(f"similarity shape {g.shape} does not match {n} runs")
    tri = g[np.triu_indices(n, 1)]
    sf = dict(subject=subject, condition=condition, band=band, m=m,
              n_runs=n, n_pairs=len(tri), sf_index=sf_index(g, threshold),
              near_zero_threshold=threshold,
              g_mean=float(np.mean(tri)) if len(tri) else np.nan,
              g_median=float(np.median(tri)) if len(tri) else np.nan)
    return dd, sf


def load_matlab_full(root, *, threshold=1e-4):
    """Read archived canonical results, including saved run similarity matrices.

    Returns ``(dd_table, sf_table, matrices)``. Cohort labels come from the
    input folders; bands and dimensions are discovered from result keys.
    """
    root = Path(root)
    dd_rows, sf_rows, matrices = [], [], {}
    for file in sorted(root.glob("*/*_ssdi_spectral_domain_canonical.mat")):
        condition = file.parent.name
        subject = file.name.removesuffix("_ssdi_spectral_domain_canonical.mat")
        canonical = loadmat(file, simplify_cells=True)["canonical"]
        for key in canonical:
            if "_canonical_band_dd_m" not in key:
                continue
            band, dim = key.rsplit("_canonical_band_dd_m", 1)
            if not dim.isdigit():
                continue
            m = int(dim)
            gkey = f"{band}_gopto_m{m}"
            if gkey not in canonical:
                continue
            values = np.atleast_1d(canonical[key])
            g = np.atleast_2d(np.asarray(canonical[gkey], float))
            dd, sf = _summarize(condition, subject, band, m, values, g, threshold)
            dd_rows.append(dd)
            sf_rows.append(sf)
            matrices[(condition, band, m, subject)] = g
    return pd.DataFrame(dd_rows), pd.DataFrame(sf_rows), matrices


def load_python_full(output, *, threshold=1e-4):
    """Summarize Python checkpoints, computing similarities from final bases."""
    output = Path(output)
    dd_rows, sf_rows, matrices = [], [], {}
    for file in sorted((output / "blocks").glob("*/*/*_m*.npz")):
        condition, subject = file.parent.parent.name, file.parent.name
        band, dim = file.stem.rsplit("_m", 1)
        m = int(dim)
        with np.load(file) as block:
            values = block["final_values"]
            bases = block["final_bases"]
        with np.load(output / "models" / condition / f"{subject}.npz") as model:
            v = model["v"]
        original = itransform_subspace(bases, v)
        g = similarity_matrix(original)
        dd, sf = _summarize(condition, subject, band, m, values, g, threshold)
        dd_rows.append(dd)
        sf_rows.append(sf)
        matrices[(condition, band, m, subject)] = g
    return pd.DataFrame(dd_rows), pd.DataFrame(sf_rows), matrices


def load_timewindow_tables(root):
    """Read archived window-level subject tables for plotting/statistics.

    The source archive is distinct from the full-recording Python parity run.
    This reader does not imply Python time-window optimization was rerun.
    """
    root = Path(root)
    stats = root / "stats"
    dd = pd.read_csv(stats / "dd_subject_long_all_windows.csv")
    sf = pd.read_csv(stats / "sf_subject_long_all_windows.csv")
    return dd, sf
