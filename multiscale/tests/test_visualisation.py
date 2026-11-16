"""Checks that publication figures remain reproducible and self-contained."""

import tempfile
import unittest
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    plt = None


@unittest.skipIf(plt is None, "install multiscale[plots] to test figures")
class FigureTests(unittest.TestCase):
    def test_export_and_notebook_style_isolation(self):
        from multiscale.visualisation import DEFAULT_COLOURS, plot_dd_grand_average

        self.assertEqual(DEFAULT_COLOURS, ("#2EC4B6", "#5E60CE"))
        table = pd.DataFrame({
            "condition": ["control", "control", "practice", "practice"],
            "band": ["alpha"] * 4,
            "dd_min": [0.2, 0.3, 0.4, 0.5],
        })
        stats = pd.DataFrame({"band": ["alpha"], "q": [0.2]})
        before = plt.rcParams["font.size"]
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "dd"
            self.assertEqual(plot_dd_grand_average(table, stats, stem), stem.with_suffix(".pdf"))
            for ext in ("pdf", "png", "svg"):
                self.assertTrue(stem.with_suffix("." + ext).is_file())
        self.assertEqual(plt.rcParams["font.size"], before)
