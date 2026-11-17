# Direct state-space fitting parity

Reference: MVGC2 commit `89be6b67f0f5e8697da333467ce41877d37e8421`.
The original `tsdata_to_ss`, `tsdata_to_ssmo`, and `demean` MATLAB source
functions were run under GNU Octave on 17 November 2026.

Fixtures use a stable two-state, three-observation innovations model, 100
burn-in samples, 600 retained samples, and nonzero observation means. The
three-trial case has horizons `(4, 3)`; the single-trial case uses `(4, 4)`.
Both explicitly fit two states and independently run reference SVC selection.

On the multitrial case, the maximum absolute differences were:

| Quantity | Python versus reference |
|:--|--:|
| Innovations covariance V | 4.45 × 10⁻¹⁶ |
| Innovations E | 1.34 × 10⁻¹⁵ |
| Transfer function, 33 frequencies | 2.89 × 10⁻¹⁵ |
| SVC selected order | 3 in both |

Regression tests cover both fixtures with absolute tolerances of `1e-12`
for innovations/covariance and `1e-11` for transfer functions and state
coordinate equivalence. Separate tests cover the pilot fit, innovation
whitening, spectral and temporal DD integration, and invalid inputs.

Regenerate with:

```bash
python multiscale/validation/generate_ss_reference.py /path/to/MVGC2
PYTHONPATH=multiscale/src python -m pytest multiscale/tests/test_ss_fit.py
```

The generator calls the original reference functions, not a second Python
implementation. The fixtures can be tested without MATLAB or Octave installed.
State estimates are identifiable only up to a similarity transform; tests
align state coordinates rather than requiring identical SVD signs.
This record verifies these numerical cases and preserves the reference's SVC
indexing. It does not assert that SVC recovers the generating dimension or
that every finite-sample fit is stable.
