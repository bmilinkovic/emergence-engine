"""Small synthetic checks for the public, data-independent workflow."""

import unittest

import numpy as np

from multiscale import run_time_changing_dd, minimum_dd_by_window


class TimeChangingDDTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.signals = np.zeros((3, 320))
        a = np.array([[.6, .12, 0], [0, .5, .1], [0, 0, .4]])
        for t in range(1, 320):
            self.signals[:, t] = a @ self.signals[:, t-1] + rng.normal(size=3)
        self.windows = {"early": (0, 160), "late": (160, 320)}
        self.common = dict(fs=64, fres=16, dimensions=[1], pmax=2,
                           criterion="aic", runs=2, pre_iters=4,
                           final_iters=4, state_space_iters=3)

    def test_both_domains_use_each_window(self):
        for domain, extra in (("temporal", {}),
                              ("spectral", {"bands": {"low": (2, 12)}})):
            with self.subTest(domain=domain):
                result = run_time_changing_dd(
                    self.signals, self.windows, domain=domain,
                    **self.common, **extra)
                rows = minimum_dd_by_window(result, domain=domain)
                self.assertEqual([row["window"] for row in rows], ["early", "late"])
                self.assertTrue(all(np.isfinite(row["dd_min"]) for row in rows))
                self.assertTrue(all(row["model_order"] > 0 for row in rows))

    def test_rejects_invalid_windows(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            run_time_changing_dd(self.signals, {"short": (0, 3)},
                                 domain="temporal", **self.common)


if __name__ == "__main__":
    unittest.main()
