# Multiscale · dynamical independence

This is the standalone multiscale package of the emergence toolkit. It computes
temporal dynamical independence and the canonical spectral objective, then
optimises low-dimensional subspaces using the SSDI two-stage procedure.

The implementation follows the numerical path used by Barnett's SSDI-1 and
MVGC2 routines. Residual normalization uses **Cholesky**, orthonormalisation
uses **SVD**, and temporal DI uses the **extended-pencil, QR-deflated,
ordered-QZ** Riccati construction. The DMT canonical spectral objective uses a
Cholesky congruence transform and Hermitian eigenvalues. These details matter
when comparing small DI values.

## Install

```bash
python -m pip install ./multiscale
```

For the read-only DMT parity script, install `./multiscale[parity]` instead.
The wheel has no MATLAB dependency and does not contain study data.
For figures and group statistics, install `./multiscale[plots]`; the DMT
figure audit needs `./multiscale[parity,plots]`.

## From time series to a VAR model

For observations in `(nodes, time, trials)` order, make two separate choices:

- **Model order criterion:** AIC, BIC, HQC, or the sequential likelihood ratio
  test (LRT).
- **Regression algorithm:** `"LWR"` or `"OLS"`. LWR is the
  Levinson–Wiggins–Robinson multichannel autoregression, using Morf's
  recursive implementation. It assumes a stable process and produces a stable
  VAR estimate. OLS is ordinary least squares, solved with rank-revealing QR;
  it can fit unstable processes. DMT used LWR for both order selection and
  fitting.

`fit_var_auto` returns the selected fit and all four candidate orders:

```python
from multiscale import fit_var_auto

selection = fit_var_auto(signals, pmax=13, criterion="bic", regmode="OLS")
print(selection.candidates)  # AIC, BIC, HQC, LRT recommendations
A, V = selection.fit.coefficients, selection.fit.covariance
```

Use a domain-appropriate order criterion; the DK5 DMT reference used LRT. Order zero means a
white-noise model and has no directed dynamics to optimize.

## Use with an already fitted VAR model

`A` has shape `(regions, regions, lags)` and `V` is the innovation covariance.
Bands are inclusive in Hz. Your sampling rate and frequency resolution must
match the model and intended spectral grid.

```python
from multiscale import random_initial_bases, run_two_stage

bands = {"low": (1, 4), "mid": (5, 12)}
starts = random_initial_bases(n=A.shape[0], m=2, runs=100, seed=42)
result = run_two_stage(A, V, fs=128, fres=128, m=2, bands=bands,
                       initial_bases=starts)
print(result["low"]["final"].values[0])
```

For a full search across an arbitrary atlas, use independent processes for
each requested dimension and band. The only dimension limit is mathematical:
`1 <= m < n_regions`. A 68-region atlas can use the same API, though its VAR
model and optimization grid will require much more compute and memory.

```python
from multiscale import run_grid_parallel

results = run_grid_parallel(
    A, V, fs=256, fres=256,
    dimensions=range(1, A.shape[0]),
    bands={"delta": (1, 4), "theta": (5, 8)},
    runs=100, workers=8, seed=42,
    progress=lambda done, total, m, band, result:
        print(f"{done}/{total}: m={m} {band}", flush=True),
)
```

The model transfer function is prepared once and shared with worker
processes. Each block has a deterministic seed across worker counts. Set
`workers` to fit the machine's memory; a large atlas and high model order
increase the memory needed by each worker.

For example, searching dimensions 1–67 in a 68-region atlas uses the same
call. Plan the grid first: its number of optimization blocks is the number
of dimensions times the number of bands, and each block runs the requested
restarts. The VAR state dimension also grows as regions times model order,
so model preparation and each worker's memory can become substantial. Start
with a small band and dimension subset to estimate the larger run's cost.

`pre` contains the sorted proxy optimization results. Representatives are
selected by SSDI-1's greedy Grassmann clustering in that sorted order.
`final` contains sorted canonical band-DD results. Its bases are in the
**decorrelated coordinates** used by the objective; `final_original_bases`
contains their inverse transform to the original signal coordinates. Clustering
uses the inverse-transformed preoptimization bases, as in DMT. To compare an entire
optimization trajectory with MATLAB, provide the same initial bases: NumPy
and MATLAB Gaussian random streams need not generate the same starts.

For temporal DI of a selected basis, call `prepare_var(A,V,fs,fres)` and pass
the returned `(A_ss,C,K)` with the basis to `iss2dd`. The temporal calculation
assumes the normalized innovation covariance is the identity. It first uses
the original ordered-QZ DARE. If the companion state's numerical conditioning
causes that solve to fail SSDI's residual threshold, it applies a Cholesky
similarity transform to unit stationary state covariance and repeats the same
DARE solve. This keeps the input-output process and DI invariant. Use
`iss2dd(..., return_diagnostics=True)` to inspect the method and residuals.

## Preprocessing and model fitting

The optimization starts from a fitted, stable VAR model. fMRI and EEG require
different preprocessing and different defensible choices of model order,
sampling rate, filtering, artifact rejection, stationarity checks, and
possibly source reconstruction. These decisions change the mathematical
process whose DI is measured. Do not copy a DMT/EEG frequency grid or order
into fMRI merely to make numbers look comparable.

## DMT parity

Two archived analyses use different sampling grids. The earlier DK68 audit
in `validation/compare_dmt.py` reads each file's saved `fs` and `fres` metadata
and compares saved canonical values; that archive used 128/128. The DK5
momax30 reference uses 256/256, LRT model selection, and dimensions 1–4.
`validation/run_dmt_full.py` reproduces the DK5 analysis from the archived raw
source-reconstructed arrays. It fits the VAR, checks that model against the
archived MATLAB fit, then runs 100 matched starts per dimension and band with
the full 10,000 iteration limits. Work is split across processes and every
block is checkpointed; `progress.log` and `status.json` show progress. The
MATLAB-generated starts are for trajectory validation only. New analyses can
use `run_grid_parallel` and its seeded Python starts without MATLAB.

For the earlier archive comparison, supply the saved result and matching model
directories:

```bash
python multiscale/validation/compare_dmt.py RESULTS_ROOT MODEL_ROOT \
  --output parity.csv
```

`--first-run` provides a fast screen; `--temporal` also checks the first
temporal DI value per band and dimension. No DMT data is included here.
The aggregate errors are in [the public parity summary](validation/PARITY_SUMMARY.md).
The read-only script can generate a detailed local CSV for per-subject review.
`validation/compare_dmt_full.py` compares each completed DK5 optimization
checkpoint with its MATLAB output, including best values and restart counts.
Pass `--temporal-repaired` to additionally compare the best Python temporal
values with repaired values at the MATLAB best bases.
The DK5 archive's temporal DARE fields are NaN because the original solver
fails its residual threshold in the ill-conditioned companion coordinates.
`validation/recompute_dmt_temporal.py` writes repaired temporal values to
local, ignored checkpoints using the covariance-balanced ordered-QZ DARE.

## Scope and provenance

The package implements the mathematical routines required for the canonical
DMT SSDI path, rather than the entirety of MVGC2. SSDI-1 is MIT licensed;
MVGC2 is GPL v3, so this package is distributed under GPL-3.0-only. See
[NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).

The temporal and spectral objective functions have been compared with saved
MATLAB results. Reproducing a complete optimization *trajectory* also
requires the same random starts, numerical library behavior, and stopping
settings; objective parity alone does not establish bitwise optimizer parity.

The original SSDI-1 frequency-integrated path is also available through
`trfun2dd`, `trfun2ddgrad`, and `opt_gd_dds_mruns`. It is distinct from the
DMT canonical **band** objective: select the routine used in the analysis you
intend to reproduce. Both paths were checked against a synthetic fixture
generated by the original MATLAB implementations.

## Group analysis and figures

The `analysis`, `statistics`, and `visualisation` modules keep data loading,
inference, and drawing separate. They work with arbitrary cohort names, band
names, and macro dimensions. The tidy input tables have one row per subject,
condition, band, and dimension (`m`), plus `dd_min` or `sf_index`.
`sf_index` is the fraction of distinct run pairs whose Grassmann distance is
below `1e-4`; it measures how often separately optimized solutions collapse
to nearly the same subspace. Keep the threshold and number of restarts fixed
across conditions when comparing it.

```python
from multiscale.statistics import paired_tests, grand_average_dd
from multiscale.visualisation import plot_dd_violin_by_macro

dd_stats = paired_tests(dd, "dd_min", "control", "meditation",
                        groups=["m", "band"], correction_groups=["m"])
grand = grand_average_dd(dd, "control", "meditation", paired=False)
plot_dd_violin_by_macro(dd, dd_stats, "figures/dd_by_macro",
                        conditions=("control", "meditation"))
```

`paired_tests` runs two-sided paired Wilcoxon tests with the approximate-p
convention, and Benjamini–Hochberg correction inside the groups you choose.
`grand_average_dd(..., paired=False)` follows the full-recording archive's
pooled rank-sum analysis; use `paired=True` for a subject-mean paired analysis.
The pooled SF two-proportion test counts run pairs and is descriptive, since
run pairs from one subject are dependent. The subject-level paired SF test is
the inferential comparison. Plotting functions save both vector PDF and PNG.

The read-only DMT figure audit in `validation/render_dmt_figures.py` accepts
the four archived roots and the Python checkpoint directory, then writes
figures, subject-level tables, statistics, and an aggregate parity JSON under
the chosen output directory. The generated study outputs are ignored by Git.
See [the DMT figure audit](validation/DMT_FIGURE_AUDIT.md) for the exact
command, figure map, and parity findings.

## Three SSDI-1 optimization stages

The original code has a fast **proxy** objective, `ddx*`, followed by two
ways to optimize actual full-spectrum DD:

1. `opt_gd_ddx_mruns`: CAK-based proxy gradient descent, used first.
2. `opt_gd_dds_mruns`: transfer-function spectral integration and gradient
   descent. This is the usual production path.
3. `opt_es_dd_mruns`: direct state-space DARE evaluation and the original
   1+1 evolution strategy. This older path is slower because it solves a DARE
   for every mutation; it is useful as an independent numerical check or
   further refinement. Its update rule and trajectory match a MATLAB fixture.

`run_full_dd` composes these methods for a fitted VAR:

```python
from multiscale import random_initial_bases, run_full_dd

starts = random_initial_bases(A.shape[0], m=2, runs=10, seed=42)
result = run_full_dd(A, V, fs=256, fres=256, initial_bases=starts,
                     final_method="both", seed=42)
print(result["spectral"].values[0], result["state_space"].values[0])
```

`"both"` starts each actual-DD method from the same clustered proxy bases.
`"spectral_then_state_space"` follows the original simulation's sequential
proxy → spectral → direct-DARE refinement. Either method can also run alone.
Set `state_space_workers` in `run_full_dd`, or `workers` in
`opt_es_dd_mruns`, to parallelize the expensive direct DARE restarts.
The DMT **band-limited canonical** analysis uses `run_two_stage` or
`run_grid_parallel` instead of this full-spectrum convenience function.
