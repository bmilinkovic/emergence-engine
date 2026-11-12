# Emergence Toolkit

*For studying what a system does together that its parts do not do alone.*

A Python home for two complementary ways of studying emergence in time series. **Synergy and Multiscale are both installable.** The wider project is still growing. 🔧

| Module | What it covers | Status |
|:--|:--|:--|
| [**Synergy**](synergy/) | ΦID computation, subject summaries, group statistics, and figures | ✅ Ready |
| [**Multiscale**](multiscale/) | VAR fitting, temporal and spectral DD, and time-changing DD across recording windows | ✅ Ready |

The project is designed for data from different modalities and cohorts. Your nodes can be fMRI regions, EEG channels, or another consistent set of signals; your groups can have whatever names your study needs.

## Get started

From the repository root, install Synergy and its analysis dependencies:

```bash
python -m pip install -e './synergy[analysis]'
```

The core ΦID engine alone has a lighter install: `python -m pip install -e ./synergy`. For accelerated pairwise calculations, add the `fast` extra. The retained AAL90 DoC figure workflow uses the `doc` extra and separate atlas assets.

To use Multiscale, install `python -m pip install ./multiscale`.
Its [guide](multiscale/README.md) shows model fitting, temporal and spectral
optimization, and recording-window analysis. Its
[quickstart notebook](multiscale/notebooks/quickstart.ipynb) uses synthetic
signals. MATLAB is not required.

### A small working example

```python
import numpy as np
from synergy import pairwise_atoms
from synergy.analysis import SubjectRecord, analyze_subjects
from synergy.statistics import infer_groups
from synergy.plotting import render_figures

rng = np.random.default_rng(7)
records = []
for group in ("control", "meditation"):
    for number in range(3):
        signal = rng.normal(size=(3, 500))  # nodes × samples
        if group == "meditation":
            signal[1] += 0.25 * signal[0]
        atoms = pairwise_atoms(signal, tau=2, methods=("mmi",))
        records.append(SubjectRecord(f"{group}_{number}", group, atoms))

analysis = analyze_subjects(
    records,
    node_labels=["Fz", "Cz", "Pz"],
    reference_group="control",
)
results = infer_groups(analysis, n_perm=999, seed=42)
print(results.group_subject[["method", "atom", "group", "group_minus_reference", "q_fdr"]])
render_figures(analysis, "figures", inference=results)
```

That example generates its own tiny dataset so you can try the workflow before connecting yours. The [quickstart notebook](synergy/notebooks/quickstart.ipynb) walks through loading prepared signals, choosing a lag, computing atoms, testing groups, and making figures. For the full API and the dataset-specific DoC commands, see the [Synergy guide](synergy/README.md).

## Bring your own signals

Synergy expects **preprocessed time series**, with one array per subject in `nodes × samples` order. Keep node order, preprocessing, and acquisition choices comparable across subjects. The lag `tau` is counted in samples: for fMRI its physical duration is `tau × TR`; for EEG it is `tau / sampling_rate` seconds.

- **fMRI:** document the parcellation, ROI order, motion and nuisance handling, temporal filtering, and whether the signals are BOLD or HRF-deconvolved.
- **EEG:** document the montage or source space, reference, bad-channel and artifact handling, filtering, epoch selection, sampling rate, and any downsampling. Consider volume conduction or source leakage when interpreting relationships between nodes.

The toolbox leaves those decisions with you. That is where a great deal of the science lives, so keep a record of them. The notebook has a practical checklist.

## Where things live

```text
emergence-toolkit/
├── synergy/             ΦID engine + general analysis, statistics, plotting
│   ├── notebooks/        runnable quickstart
│   └── src/synergy/      installable Python package
└── multiscale/          standalone temporal, spectral, and time-changing DD
```

Synergy computes the 16 ΦID atoms and offers MMI/CCS pairwise matrices. Its general analysis accepts arbitrary group and node labels; the original DoC analysis and cortical figures remain available as a dataset-specific preset. The ΦID computation was originally extracted from Pedro Mediano's MATLAB toolboxes.

## License

Synergy: [MIT](synergy/LICENSE). Multiscale: [GPL-3.0](multiscale/LICENSE)
because its numerical path adapts MVGC2. © 2026 Borjan Milinkovic.
