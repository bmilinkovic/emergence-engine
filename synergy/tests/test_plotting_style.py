"""Regression checks for semantic colors and figure export."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from synergy.plotting.general import COLOURS, plot_global_atoms, save_figure
except ImportError:
    plt = None


@unittest.skipIf(plt is None, "install synergy[analysis] to test figures")
class FigureTests(unittest.TestCase):
    def test_atom_palette_and_style_isolation(self):
        self.assertEqual(COLOURS["rtr"], "#2E5EAA")
        self.assertEqual(COLOURS["sts"], "#D43D2A")
        subject = pd.DataFrame([
            {"method": "mmi", "atom": atom, "group": group, "mean_edge": value}
            for atom, value in (("rtr", 0.2), ("sts", 0.3), ("balance", 0.1))
            for group in ("control", "practice")
        ])
        result = SimpleNamespace(subject=subject, methods=("mmi",),
                                 group_order=("control", "practice"))
        before = plt.rcParams["font.size"]
        fig = plot_global_atoms(result)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_figure(fig, Path(directory) / "atoms")
            self.assertEqual({p.suffix for p in paths}, {".png", ".pdf", ".svg"})
            self.assertTrue(all(p.is_file() for p in paths))
        plt.close(fig)
        self.assertEqual(plt.rcParams["font.size"], before)
