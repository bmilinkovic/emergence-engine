# Synergy

Synergy is an installable Python toolbox for Phi_ID computation, subject-level analysis, statistical inference, and figures. It accepts arbitrary groups, region or channel labels, and node counts. The Phi_ID computation was originally extracted from Pedro Mediano's MATLAB toolboxes. The core runs without MATLAB or Java. MMI and CCS values use natural logarithms (nats); shared-exclusions values use bits.

## Install

```bash
python -m pip install './synergy[analysis]'
```

The basic Phi_ID engine needs only NumPy and SciPy (`python -m pip install ./synergy`). The optional `fast` extra adds Numba for larger pairwise analyses. The optional `doc` extra adds atlas and surface dependencies for the retained AAL90 DoC figure workflow.

## Start with your data

Open [the quickstart notebook](notebooks/quickstart.ipynb). It loads a small synthetic dataset, computes Phi_ID matrices, summarises subjects, tests arbitrary groups, and creates figures. Replace its arrays with your **preprocessed** time series. Each array is `nodes × samples`, and all subjects must use the same node order.

```python
from synergy import pairwise_atoms
from synergy.analysis.workflow import SubjectRecord, analyse_subjects
from synergy.statistics.inference import infer_groups
from synergy.plotting.general import render_figures

atoms = pairwise_atoms(prepared_signal, tau=2, methods=("mmi", "ccs"))
record = SubjectRecord("participant_01", "meditation", atoms)
analysis = analyse_subjects(
    records, node_labels=channel_names,
    group_order=["control", "meditation"], reference_group="control",
)
inference = infer_groups(analysis, n_perm=10_000, seed=42)
figures = render_figures(analysis, "figures", inference=inference)
```

`records` is a list of `SubjectRecord` objects. Each record can also contain optional matched connectivity matrices, such as FC, SC, or EEG coherence, via `connectivity={"name": matrix}`. `node_systems` is optional; pass one system label per node to add system-level summaries and violin plots. The general analysis is in `synergy.analysis.workflow`, inference in `synergy.statistics.inference`, and figures in `synergy.plotting.general`.

`analyse_subjects` is the preferred spelling; `analyze_subjects` remains available for existing scripts.

The general renderer adapts to the available inputs. It can create global atom, paired atom, graph metric, node gradient, system gradient, regional effect, and connectivity similarity panels. It uses node heatmaps for spatial summaries, so EEG does not require a brain atlas. The dataset-specific cortical surface functions remain available in `synergy.plotting` for compatible fMRI data.

### Preprocessing matters

The toolbox analyses the signal you give it; it does not clean raw fMRI or EEG recordings. Keep preprocessing comparable across groups and record the pipeline and physical lag.

| fMRI | EEG |
|---|---|
| State parcellation, ROI order, TR, motion handling, nuisance regression, censoring, and temporal filtering. | State sensor/source choice, montage, reference, sampling rate, bad-channel handling, line-noise/artefact removal, filtering, and epoch selection. |
| State whether data are BOLD or HRF-deconvolved; haemodynamic smoothing and regional HRF differences affect lagged measures. | Consider volume conduction and source leakage; downsampling changes the lag represented by `tau`. |
| Interpret `tau` in TR samples. | Interpret `tau` in samples after preprocessing (`tau / sampling_rate` seconds). |

For both modalities, remove nonfinite or near-constant channels, use comparable windows and acquisition conditions across subjects, and check that the window is reasonably stationary and long enough for covariance estimation. The notebook walks through these checks.

## Phi_ID engine

`phiid_full` and `phiid_full_discrete` compute all 16 atoms. They accept a `variables × time` array or four explicit past/source and future/target arrays `(X1, X2, Y1, Y2)`. For more than two variables, the engine selects a minimum-information bipartition automatically; pass `partition=([0, 1], [2])` to choose one explicitly. The split is returned as `result.partition`.

| ΦID measure | Discrete states · `phiid_full_discrete` | Continuous signals · `phiid_full` |
|:--|:--:|:--:|
| Minimum mutual information (MMI) | ✓ `measure="MMI"` | ✓ `measure="MMI"` |
| Common change in surprisal (CCS) | ✓ `measure="CCS"` | ✓ `measure="CCS"` |
| Shared exclusions (`Iτ-sx`) | ✓ `measure="SX"` | ✗ |
| Proposed two-sided Gaussian SX extension | ✗ | ✓ `measure="SX_GAUSSIAN"` · experimental |

✓ means implemented; ✗ means not implemented for that input type. Discrete
MMI and CCS use empirical probabilities; continuous MMI and CCS use a fitted
Gaussian model. They return nats. Discrete SX uses integer-coded states and
returns bits. The Gaussian SX extension also returns bits, requires two scalar
past and two scalar future variables, and remains an experimental proposal.
The two SX rows are distinct measures.

```python
from synergy import phiid_full, phiid_full_discrete

# Continuous, Gaussian: two channels × time samples
gaussian = phiid_full(signals, tau=1, measure="CCS", return_local=True)
gaussian_sx_experimental = phiid_full(
    signals, tau=1, measure="SX_GAUSSIAN"
)

# Discrete: two integer-coded channels × time samples
discrete = phiid_full_discrete(states, tau=1, measure="MMI")
shared_exclusions = phiid_full_discrete(states, tau=1, measure="SX")
```

Each result has `atoms` for the 16 average contributions; `return_local=True`
also supplies `local_atoms` for each lagged sample. For a many-node continuous
recording, `pairwise_atoms` computes MMI and CCS matrices (RTR and STS by
default, or all 16 directed atoms with `all_atoms=True`). It does not compute
either SX measure.

The selector evaluates every nontrivial split `A|B`: delayed mutual information for the full system minus the two within-part delayed mutual informations, normalised by the smaller past-state entropy. Gaussian entropy uses the lagged sample covariance; discrete entropy uses empirical state frequencies. The search has `2^(D-1)-1` candidates and defaults to at most 12 variables.

`pairwise_atoms` returns symmetric `rtr` and `sts` matrices for MMI and CCS for any node count. Set `all_atoms=True` for all 16 directed matrices per estimator. Entry `[i, j]` treats node `i` as `x` and node `j` as `y`. Constant or nonfinite channels must be filtered before calling it.

The 16 keys are `rtr, rtx, rty, rts, xtr, xtx, xty, xts, ytr, ytx, yty, yts, str, stx, sty, sts`. Atoms can be negative. Continuous inputs use fitted Gaussian local information; discrete inputs use empirical probabilities. `return_local=True` returns one value per lagged sample per atom.

### Shared-exclusions PhiID (Varley 2023)

Use `phiid_full_discrete(X, tau=1, measure="SX", return_local=True)` for the
pointwise shared probability mass exclusions measure, also called
`I_tau-sx`. It returns the 16 atoms in **bits**, with one local value per
lagged sample when requested. The estimator uses empirical frequencies without
smoothing. Input must already be integer-coded discrete states; it does not
silently bin continuous EEG or fMRI signals. Choose and report a binning scheme
before calling `SX` on continuous recordings. With many states relative to
samples, empirical probabilities are sparse and estimates can be biased.
The established Gaussian API supports `CCS` and `MMI`. A separate experimental
continuous extension is described below.

```python
from synergy import phiid_full_discrete

# spikes: integer array with shape (2, samples), e.g. 0/1 spike bins
result = phiid_full_discrete(spikes, tau=1, measure="SX", return_local=True)
print(result.atoms["sts"], result.local_atoms["sts"])
```

Reference: [Varley (2023), *Decomposing past and future*](https://doi.org/10.1371/journal.pone.0282950).

### Experimental Gaussian shared-exclusions PhiID

`phiid_full(X, measure="SX_GAUSSIAN", return_local=True)` computes a proposed
**two-sided** continuous shared-exclusions decomposition for two scalar signals
in `X` (shape `2 × samples`). It also accepts four explicit scalar arrays
`(X1, X2, Y1, Y2)`. This uses a fitted joint Gaussian density, standardizes
each variable to unit variance as a fixed scale convention, computes 16 local
double redundancies in bits, and inverts the PhiID lattice. It rejects
singular covariance estimates and multivariate groups.

This is **not a published Gaussian version of Varley's `I_tau-sx`**. It applies
the continuous single-target disjunction rule of [Ehrlich et al. (2024)](https://doi.org/10.1103/PhysRevE.110.014115)
to *both* the past and future sides. That symmetric extension is a research
proposal implemented here, with local MI reconstruction and single-target
reduction tests. Density scale choices affect SX; the explicit standardization
convention must be reported. Treat the outputs as exploratory, especially on
short or nonstationary windows.

```python
from synergy import phiid_full

# signals: two scalar continuous channels, shape (2, samples)
result = phiid_full(signals, tau=1, measure="SX_GAUSSIAN", return_local=True)
print(result.atoms["sts"], result.local_atoms["sts"])
```

## Dataset-specific DoC figures

The existing AAL90 analysis and plotting functions remain available for repaired DoC matrices. Install `./synergy[doc]` and use `synergy-analysis analyse` or `synergy-analysis render`; these commands expect the original repaired matrix layout, manifest, AAL atlas, and FreeSurfer fsaverage. They are a preset for that data contract. The general API above has no cohort or atlas requirement.

## Licence

MIT Licence, copyright © 2026 Borjan Milinkovic. See [LICENSE](LICENSE).
