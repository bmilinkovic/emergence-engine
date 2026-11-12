# DMT MATLAB parity summary

The archived canonical run contains 38 subject/condition result files. Its
saved metadata specifies `fs=128` and `fres=128`, which the validator uses
instead of later configuration defaults.

Recomputing the canonical spectral band objective at **all 63,422 saved
final-restart bases** gave a maximum absolute error of **1.43 × 10⁻¹⁴**.
Recomputing temporal DI through the extended-pencil ordered-QZ Riccati path
on one saved basis per file (38 comparisons) gave a maximum absolute error of
**1.08 × 10⁻¹²**.

The included synthetic MATLAB fixture also checks LWR and OLS VAR fitting,
the four model order criteria, state-space conversion, the CAK proxy, canonical
gradient, original integrated spectral objective, both optimizers, and
temporal DI. Objective values agree at floating point precision. Near a
stopping tolerance, a few optimizer iterations can differ across numerical
libraries even when the final objectives and subspaces agree closely.

The validation script accepts paths to the archived results and models and
can write a detailed CSV locally. No participant data or identifiers are
included in this repository.

**Scope of this validation:** the 63,422 spectral calculations evaluate the
objective at bases saved by MATLAB. They are not 63,422 independently rerun
optimizations. The temporal comparisons also evaluate saved bases. A complete
100-start preoptimization plus clustered final optimization was run for one
real DMT block as a runtime and pipeline check. It took 35.7 seconds on the
test machine: 100 proxy runs and 76 clustered final runs. The best proxy and
final objectives differed from MATLAB by 2.51 × 10⁻¹⁴ and 6.92 × 10⁻¹⁶,
respectively. MATLAB retained 69 clusters for that block. Clustering uses a
very strict 10⁻¹⁰ Grassmann threshold, and small optimizer differences near
that threshold changed the number of retained representatives. When given
MATLAB's saved preoptimization bases, Python selected exactly the same 69
representatives; the distance matrices differed by at most 4.44 × 10⁻¹⁶.
The full DMT
cohort has not been rerun end to end with the Python optimizer.
