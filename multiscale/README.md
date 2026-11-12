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

## From time series to a VAR model

For observations in `(nodes, time, trials)` order, choose the model order
criterion explicitly: **AIC**, **BIC**, **HQC**, or the sequential likelihood
ratio test (**LRT**). `fit_var_auto` uses MVGC2's original LWR fitting recursion and
covariance convention, and returns all four candidate orders for inspection:

```python
from multiscale import fit_var_auto

selection = fit_var_auto(signals, pmax=13, criterion="bic")
print(selection.candidates)  # AIC, BIC, HQC, LRT recommendations
A, V = selection.fit.coefficients, selection.fit.covariance
```

Use a domain-appropriate order criterion; DMT used AIC. Order zero means a
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
assumes the normalized innovation covariance is the identity.

## Preprocessing and model fitting

The optimization starts from a fitted, stable VAR model. fMRI and EEG require
different preprocessing and different defensible choices of model order,
sampling rate, filtering, artifact rejection, stationarity checks, and
possibly source reconstruction. These decisions change the mathematical
process whose DI is measured. Do not copy a DMT/EEG frequency grid or order
into fMRI merely to make numbers look comparable.

## DMT parity

`validation/compare_dmt.py` compares all saved canonical band values against
the archived MATLAB `.mat` files, per condition, subject, band, dimension,
and restart. It reads each result's **saved `fs` and `fres` metadata**. This
matters: the archived run used 128/128, while a later configuration file says
256/256. Supply the result directory and the matching model directory:

```bash
python multiscale/validation/compare_dmt.py RESULTS_ROOT MODEL_ROOT \
  --output parity.csv
```

`--first-run` provides a fast screen; `--temporal` also checks the first
temporal DI value per band and dimension. No DMT data is included here.
The aggregate errors are in [the public parity summary](validation/PARITY_SUMMARY.md).
The read-only script can generate a detailed local CSV for per-subject review.

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
