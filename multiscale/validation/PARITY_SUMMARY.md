# DMT MATLAB parity summary

## DK5/256 Hz, momax30: full raw-source rerun

The reference has 38 rest/DMT recordings, five regions, LRT model-order
selection up to 30 lags, and macro dimensions 1–4. Python fitted a VAR from
every raw source-reconstructed recording. All 38 selected orders matched
MATLAB; maximum absolute coefficient and innovation-covariance differences
were **1.61 × 10⁻¹⁰** and **5.25 × 10⁻¹⁸**, respectively.

The full Python optimization used 100 MATLAB-matched starts per dimension
and band, 10,000-iteration limits for both stages, eight worker processes,
and checkpointed all **760 blocks**. Each of the 38 files contributed 20
blocks. Subject-by-subject comparison of each block's best solution found:

| Measure | Maximum absolute difference |
| --- | ---: |
| Best canonical band-DD value | **2.22 × 10⁻¹¹** |
| Best subspace projector (Frobenius norm) | **3.51 × 10⁻⁶** |
| Best proxy DDx value | **1.93 × 10⁻⁶** |

All 760 best canonical values differ by less than 10⁻⁸. Re-evaluating the
canonical objective at **all 39,998 MATLAB-saved final bases** gave maximum
absolute error **1.06 × 10⁻¹¹**.

The complete restart lists are not identical: 672 of 760 blocks retained a
different number of final cluster representatives, with count differences
from −18 to +12. The Grassmann clustering threshold is 10⁻¹⁰, so small
numerical differences in preoptimization trajectories can change these
counts. Applying Python's greedy cluster selection to MATLAB's saved distance
matrices reproduced the archived representatives in **760/760 blocks**. The
best DD and best subspace nevertheless agree closely. This establishes
best-solution parity, not bitwise trajectory or full-file parity.

### Temporal DARE repair

All **39,998** temporal DARE values in the archived DK5 MATLAB files were
NaN. The original ordered-QZ DARE check rejected the ill-conditioned VAR
companion-state realization. Python now first tries that original solve, then
uses a Cholesky similarity transform based on the stationary state covariance
and repeats the **same ordered-QZ solver** if the residual is too large. The
transform preserves the model's input-output process and the temporal DD.

Recomputing temporal DD at every saved final basis produced **39,998/39,998
finite values**; the largest DARE residual was **1.24 × 10⁻¹¹**. For one saved
basis in each of the 760 blocks, these values agreed with a 4096-bin
full-spectrum calculation within **1.15 × 10⁻⁹**. The archive has no finite
MATLAB temporal value to compare with, so this is an independent mathematical
cross-check rather than MATLAB numeric parity.

Recomputing temporal DD at each independently optimized Python best basis
and comparing it with the repaired value at the corresponding MATLAB best
basis gave maximum absolute difference **2.77 × 10⁻⁸** across all 760 blocks.
All 760 temporal solves were finite and had residual below **1.04 × 10⁻¹¹**.

The direct state-space 1+1 evolution-strategy optimizer was separately
matched to a synthetic MATLAB trajectory, including every step-size and DD
history value. The transfer-function gradient optimizer and proxy optimizer
also match synthetic MATLAB fixtures. The package's 13 direct tests pass,
and its standalone wheel builds successfully. A 68-region synthetic smoke
run confirmed that the parallel grid API is not restricted to DK5.

Detailed subject-level checkpoints and CSVs remain local and ignored by Git.
No participant data or identifiers are included in this repository.

## Earlier DK68/128 Hz archive

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
The full DK68 cohort was not rerun end to end with the Python optimizer.
