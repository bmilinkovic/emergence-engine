# Attribution

This package adapts mathematical routines from:

- Lionel Barnett's **MVGC2** toolbox (GPL v3), including the extended-pencil
  QZ construction for `mdare`, `var_to_ss`, `ss2trfun`, and `sfreqs`.
- Lionel Barnett's **SSDI-1** toolbox (MIT), including the CAK proxy,
  Grassmann optimization, and clustering procedure.
- The DMT project's canonical band objective and optimizer.
- Andrew Knyazev and Rico Argentati's `subspacea` small-angle method (BSD),
  used in the clustering distance.

See the source MATLAB files and license notices in their original toolboxes.
The packaged Python code is GPL-3.0-only because of the MVGC2-derived path.
