# Synergy

Synergy is an installable Python toolbox for Phi_ID computation, subject-level analysis, statistical inference, and figures. It accepts arbitrary groups, region or channel labels, and node counts. The Phi_ID computation was originally extracted from Pedro Mediano's MATLAB toolboxes. The core runs without MATLAB or Java. Information values use natural logarithms (nats).

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

The selector evaluates every nontrivial split `A|B`: delayed mutual information for the full system minus the two within-part delayed mutual informations, normalised by the smaller past-state entropy. Gaussian entropy uses the lagged sample covariance; discrete entropy uses empirical state frequencies. The search has `2^(D-1)-1` candidates and defaults to at most 12 variables.

`pairwise_atoms` returns symmetric `rtr` and `sts` matrices for MMI and CCS for any node count. Set `all_atoms=True` for all 16 directed matrices per estimator. Entry `[i, j]` treats node `i` as `x` and node `j` as `y`. Constant or nonfinite channels must be filtered before calling it.

The 16 keys are `rtr, rtx, rty, rts, xtr, xtx, xty, xts, ytr, ytx, yty, yts, str, stx, sty, sts`. Atoms can be negative. Continuous inputs use fitted Gaussian local information; discrete inputs use empirical probabilities. `return_local=True` returns one value per lagged sample per atom.

## Dataset-specific DoC figures

The existing AAL90 analysis and plotting functions remain available for repaired DoC matrices. Install `./synergy[doc]` and use `synergy-analysis analyse` or `synergy-analysis render`; these commands expect the original repaired matrix layout, manifest, AAL atlas, and FreeSurfer fsaverage. They are a preset for that data contract. The general API above has no cohort or atlas requirement.

## Licence

MIT Licence, copyright © 2026 Borjan Milinkovic. See [LICENSE](LICENSE).
