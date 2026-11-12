# DK5 DMT figure and statistics audit

Run date: 2026-11-12. The local output is
`multiscale/outputs/dmt_figures_parity_20261112/` (ignored by Git). Every
figure is saved as PDF and PNG. This document contains aggregate results only;
the subject tables and study figures are not distributed with the package.

## Reproduce the local audit

From the repository root, after installing `./multiscale[parity,plots]`:

```bash
python multiscale/validation/render_dmt_figures.py \
  --matlab-full momax13=/Volumes/ex_data/data_5meoDMT/results/ssdi_dk5_256hz_lrt_momax13 \
  --matlab-full momax30=/Volumes/ex_data/data_5meoDMT/results/ssdi_dk5_256hz_lrt_momax30 \
  --timewindow timewindow_momax13=/Volumes/ex_data/data_5meoDMT/results/ssdi_dk5_256hz_timewindows_lrt_momax13 \
  --timewindow timewindow_momax30=/Volumes/ex_data/data_5meoDMT/results/ssdi_dk5_256hz_timewindows_lrt_momax30 \
  --python-full multiscale/outputs/dmt_dk5_momax30 \
  --output multiscale/outputs/dmt_figures_parity_20261112
```

The five full-recording figures are written to each of `momax13/figures/`,
`momax30/figures/`, and `python_momax30/figures/`:

| Figure stem | Function | What it shows |
| --- | --- | --- |
| `canonical_dd_min_violin_by_macro` | `plot_dd_violin_by_macro` | DD distributions by band, cohort, and macro dimension, with paired-test stars |
| `canonical_dd_min_grand_average` | `plot_dd_grand_average` | DD pooled over dimensions, by band and cohort |
| `sf_index_paired_main` | `plot_sf_index` | Subject-level SF trajectories and paired-test stars |
| `sf_index_pooled_main` | `plot_sf_index` | The same SF trajectories with pair-count pooled-test stars |
| `sf_similarity_matrices_supplementary` | `plot_similarity_matrices` | Mean top-40 run-distance matrices by dimension, band, and cohort |

The six time-window figures are written to both `timewindow_momax13/figures/`
and `timewindow_momax30/figures/`:

| Figure stem | Function | What it shows |
| --- | --- | --- |
| `grand_average_DD_composite_timewindows` | `plot_timewindow_dd` | Baseline versus each window, across bands |
| `SF_index_4x5_timewindow_band_composite` | `plot_timewindow_sf` | SF by macro dimension for each window and band |
| `dd_state_trajectory_by_band` | `plot_state_trajectory` | Subject DD paths and group means across states |
| `dd_pairwise_state_heatmap_by_band` | `plot_pairwise_heatmap` | All paired state DD differences and significance |
| `sf_state_trajectory_by_band` | `plot_state_trajectory` | Subject SF paths and group means across states |
| `sf_pairwise_state_heatmap_by_band` | `plot_pairwise_heatmap` | All paired state SF differences and significance |

There are 27 PDFs and 27 PNGs in total. The plotting modules are study
independent; the command above supplies the DMT-specific archive names.

## Parity

The two full-recording MATLAB archives each contain 760 subject × condition ×
band × dimension blocks. Reconstructing minimum DD and SF from their saved MAT
results matches their archived subject CSVs: maximum absolute differences are
below `1e-16` for DD and `1.2e-16` for SF. All four full-recording statistics
tables in each archive match the independently recomputed p and BH-adjusted q
values to at most `1.7e-16`.

For each time-window archive, the three baseline-versus-window tables and four
all-state pairwise tables have exactly the same p and adjusted q values as
recomputed here. The correction families are: DD across bands within each
window × dimension, SF across dimensions within each window × band; all-state
pairwise DD across bands within a dimension and SF across dimensions within a
band. The pairwise grand comparisons correct across bands within each state
pair. These figures use the archived time-window subject data; time-window
optimizations have not been rerun in Python.

For the independently optimized Python **momax30** full-recording run, all 760
minimum DD values match MATLAB to a maximum absolute difference of
`2.22e-11` (median `6.58e-13`). Its 20 paired DD and five grand DD tests have
identical p, q, and significance calls. The SF formula and distance calculation
match MATLAB when given the same saved bases, but the independent optimizer
does not always return the same set or count of restart solutions. The Python
SF index differs by up to `0.1041` (median `0.0127`) across blocks. One of 20
paired SF significance calls changes: delta band, `m=3`, archived `q=0.0703`
versus Python `q=0.0281`. The pooled SF significance calls agree. **DD is at
numeric parity; independently optimized SF is not yet at subject-level
parity.** Matching SF requires matching the entire restart solution set, not
only its best DD value.

The machine-readable aggregate is in the local `parity_summary.json`. The
full optimization audit is in [PARITY_SUMMARY.md](PARITY_SUMMARY.md).
