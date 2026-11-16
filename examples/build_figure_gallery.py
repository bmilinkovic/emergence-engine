"""Build synthetic figure examples; no values represent experimental results.

Run after installing both plotting extras from the repository root. Outputs
are written under the ignored figures/gallery directory.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multiscale.visualisation import plot_dd_grand_average, plot_time_changing_dd
from synergy.analysis.workflow import SubjectRecord, analyse_subjects
from synergy.plotting.general import render_figures

out = Path(__file__).resolve().parents[1] / "figures" / "gallery"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

dd = pd.DataFrame([
    {"condition": cond, "band": band, "dd_min": 0.3 + 0.12 * i + 0.08 * j + rng.normal(0, 0.07)}
    for i, cond in enumerate(("control", "practice"))
    for j, band in enumerate(("alpha", "beta", "gamma"))
    for _ in range(12)
])
stats = pd.DataFrame({"band": ("alpha", "beta", "gamma"), "q": (1.0, 1.0, 1.0)})
before = plt.rcParams["font.size"]
plot_dd_grand_average(dd, stats, out / "multiscale_dd")
assert plt.rcParams["font.size"] == before

windows = pd.DataFrame([
    {"window": window, "band": "alpha", "dimension": m,
     "dd_min": 0.1 * m + 0.03 * wi + rng.normal(0, 0.02)}
    for wi, window in enumerate(("before", "during", "after"))
    for m in (1, 2, 3)
])
plot_time_changing_dd(windows, out / "multiscale_time")

records = []
for group in ("control", "practice"):
    for n in range(5):
        atoms = {}
        for method in ("mmi", "ccs"):
            rtr = rng.uniform(0.04, 0.22, (5, 5))
            sts = rng.uniform(0.07, 0.28, (5, 5))
            rtr = (rtr + rtr.T) / 2
            sts = (sts + sts.T) / 2
            np.fill_diagonal(rtr, 0)
            np.fill_diagonal(sts, 0)
            atoms[method] = {"rtr": rtr, "sts": sts}
        records.append(SubjectRecord(f"{group}_{n}", group, atoms))
result = analyse_subjects(records, node_labels=("Fz", "Cz", "Pz", "Oz", "Tz"),
                          node_systems=("frontal", "central", "parietal", "occipital", "temporal"),
                          group_order=("control", "practice"))
render_figures(result, out / "synergy", formats=("png", "pdf", "svg"))
assert plt.rcParams["font.size"] == before
print("Generated preview figures in", out)
