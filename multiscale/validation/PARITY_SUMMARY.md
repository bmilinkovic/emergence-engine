# DMT MATLAB parity summary

The archived canonical run contains 38 subject/condition result files. Its
saved metadata specifies `fs=128` and `fres=128`, which the validator uses
instead of later configuration defaults.

Recomputing the canonical spectral band objective at **all 63,422 saved
final-restart bases** gave a maximum absolute error of **1.43 × 10⁻¹⁴**.
Recomputing temporal DI through the extended-pencil ordered-QZ Riccati path
on one saved basis per file (38 comparisons) gave a maximum absolute error of
**1.08 × 10⁻¹²**.

The included synthetic MATLAB fixture also checks LWR VAR fitting, the four
model order criteria, state-space conversion, the CAK proxy, canonical
gradient, original integrated spectral objective, both optimizers, and
temporal DI. Objective values agree at floating point precision. Near a
stopping tolerance, a few optimizer iterations can differ across numerical
libraries even when the final objectives and subspaces agree closely.

The validation script accepts paths to the archived results and models and
can write a detailed CSV locally. No participant data or identifiers are
included in this repository.
