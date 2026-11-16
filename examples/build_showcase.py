"""Rebuild the README figures from a reproducible six-node toy VAR(2) model.

The signals are simulated and all DD/Phi_ID values are actually computed by
Emergence Engine. These figures demonstrate analysis outputs, not empirical
results. Modest optimisation settings keep the example runnable on a laptop.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multiscale import fit_var_auto, run_grid_parallel, run_time_changing_dd
from multiscale.analysis import summarise_grid_results, summarise_window_results
from multiscale.visualisation import (
    plot_dd_grand_average,
    plot_sf_index,
    plot_time_changing_dd,
)
from synergy import pairwise_atoms
from synergy.analysis.workflow import SubjectRecord, analyse_subjects
from synergy.plotting.general import (
    plot_atom_pairs,
    plot_global_atoms,
    plot_node_gradients,
    save_figure,
)


OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "showcase"
OUTPUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(1406)
FS = 64
FRES = 32
BANDS = {"theta": (2, 7), "alpha": (8, 13), "beta": (14, 24)}
DIMENSIONS = (1, 2, 3)
NODE_LABELS = ("Fz", "FCz", "Cz", "Pz", "POz", "Oz")


def transition(coupling: float) -> tuple[np.ndarray, np.ndarray]:
    """Stable, noncommuting lag matrices with variable cross coupling."""
    a1 = np.diag([1.15, 1.08, .78, .62, .38, .30])
    a1[0, 1], a1[1, 2], a1[3, 4] = .14, .10, .09
    a1[2, 3], a1[4, 1] = coupling, coupling * .65
    a2 = np.diag([-.62, -.55, -.45, -.38, -.32, -.29])
    a2[2, 0], a2[5, 3] = .14, .06
    a2[4, 2] = -coupling * .7
    companion = np.block([[a1, a2], [np.eye(6), np.zeros((6, 6))]])
    assert max(abs(np.linalg.eigvals(companion))) < 1
    return a1, a2


def simulate(couplings: list[float], length: int, noise: np.ndarray | None = None) -> np.ndarray:
    """Produce one or more contiguous phases from the stated VAR matrices."""
    total = length * len(couplings)
    if noise is None:
        noise = RNG.normal(size=(len(NODE_LABELS), total))
    x = np.zeros_like(noise)
    matrices = [transition(c) for c in couplings]
    for t in range(2, total):
        a1, a2 = matrices[t // length]
        x[:, t] = a1 @ x[:, t-1] + a2 @ x[:, t-2] + noise[:, t]
    return x


def multiscale_figures(signals: dict[tuple[str, int], np.ndarray]) -> None:
    dd_tables, sf_tables = [], []
    for (condition, subject), x in signals.items():
        fit = fit_var_auto(x, pmax=4, criterion="aic", regmode="LWR")
        grid = run_grid_parallel(
            fit.fit.coefficients, fit.fit.covariance, FS, FRES,
            DIMENSIONS, BANDS, runs=6, seed=subject + 10,
            workers=1, pre_iters=30, final_iters=30,
        )
        dd, sf, _ = summarise_grid_results(
            grid, subject=f"S{subject+1:02d}", condition=condition,
            threshold=.2,
        )
        dd_tables.append(dd)
        sf_tables.append(sf)

    dd = pd.concat(dd_tables, ignore_index=True)
    sf = pd.concat(sf_tables, ignore_index=True)
    no_tests = pd.DataFrame({"band": list(BANDS), "q": 1.0})
    plot_dd_grand_average(dd, no_tests, OUTPUT / "multiscale-dd",
                          conditions=("low coupling", "high coupling"))
    plot_sf_index(sf, pd.DataFrame([dict(band=b, m=m, q=1.0)
                  for b in BANDS for m in DIMENSIONS]), OUTPUT / "multiscale-sf",
                  conditions=("low coupling", "high coupling"))

    changing = simulate([.04, .11, .18], length=360)
    fitted = run_time_changing_dd(
        changing, {"early": (0, 360), "middle": (360, 720),
                   "late": (720, 1080)},
        domain="spectral", fs=FS, fres=FRES, dimensions=DIMENSIONS,
        bands={"theta": BANDS["theta"], "beta": BANDS["beta"]},
        pmax=4, criterion="aic", runs=6, seed=29, workers=1,
        pre_iters=30, final_iters=30,
    )
    window_table = summarise_window_results(fitted, domain="spectral")
    plot_time_changing_dd(window_table, OUTPUT / "multiscale-windows")


def synergy_figures(signals: dict[tuple[str, int], np.ndarray]) -> None:
    records = []
    for (condition, subject), x in signals.items():
        atoms = pairwise_atoms(x, tau=1, methods=("mmi", "ccs"))
        records.append(SubjectRecord(
            subject_id=f"{condition}_{subject+1:02d}", group=condition, atoms=atoms,
        ))
    result = analyse_subjects(records, node_labels=NODE_LABELS,
                              group_order=("low coupling", "high coupling"),
                              reference_group="low coupling")
    for name, figure in (
        ("synergy-atoms", plot_global_atoms(result)),
        ("synergy-paired", plot_atom_pairs(result)),
        ("synergy-nodes", plot_node_gradients(result)),
    ):
        save_figure(figure, OUTPUT / name, formats=("png", "pdf"))
        plt.close(figure)


if __name__ == "__main__":
    # Paired conditions reuse each subject's innovations, changing only A.
    signals = {}
    for subject in range(8):
        noise = RNG.normal(size=(len(NODE_LABELS), 720))
        for condition, coupling in (("low coupling", .04),
                                    ("high coupling", .18)):
            signals[(condition, subject)] = simulate([coupling], 720, noise)
    multiscale_figures(signals)
    synergy_figures(signals)
    # Matplotlib leaves spaces at the end of multiline SVG path data.
    for svg in OUTPUT.glob("multiscale-*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(f"Built six VAR-derived figure families in {OUTPUT}")
