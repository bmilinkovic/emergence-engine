"""Checks for the proposed two-sided Gaussian SX research estimator."""

import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from synergy import ATOM_NAMES, phiid_full
from synergy.phiid import PHIID_LATTICE


class GaussianSharedExclusionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2468)
        independent = rng.standard_normal((4, 1500))
        transform = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 1.0, 0.0, 0.0],
            [0.7, 0.3, 1.0, 0.0],
            [0.1, 0.5, 0.3, 1.0],
        ])
        cls.data = transform @ independent

    def test_sums_to_gaussian_local_mutual_information(self):
        result = phiid_full(*self.data, measure="SX_GAUSSIAN", return_local=True)
        z = (self.data - self.data.mean(axis=1, keepdims=True)) / self.data.std(axis=1, keepdims=True)
        covariance = np.cov(z)

        def log_density(indices):
            return multivariate_normal.logpdf(
                z[list(indices)].T,
                mean=np.zeros(len(indices)),
                cov=covariance[np.ix_(indices, indices)],
            )

        direct_mi = (
            log_density((0, 1, 2, 3))
            - log_density((0, 1))
            - log_density((2, 3))
        ) / np.log(2)
        local_sum = sum(result.local_atoms.values())
        np.testing.assert_allclose(local_sum, direct_mi, atol=1e-12)
        self.assertAlmostEqual(sum(result.atoms.values()), direct_mi.mean())

    def test_source_redundancy_to_joint_target_matches_continuous_pid_formula(self):
        result = phiid_full(*self.data, measure="SX_GAUSSIAN", return_local=True)
        z = (self.data - self.data.mean(axis=1, keepdims=True)) / self.data.std(axis=1, keepdims=True)
        covariance = np.cov(z)

        def log_density(indices):
            return multivariate_normal.logpdf(
                z[list(indices)].T,
                mean=np.zeros(len(indices)),
                cov=covariance[np.ix_(indices, indices)],
            )

        direct = (
            logsumexp(np.vstack([log_density((0, 2, 3)), log_density((1, 2, 3))]), axis=0)
            - logsumexp(np.vstack([log_density((0,)), log_density((1,))]), axis=0)
            - log_density((2, 3))
        ) / np.log(2)
        local = np.vstack([result.local_atoms[name] for name in ATOM_NAMES])
        np.testing.assert_allclose((PHIID_LATTICE @ local)[3], direct, atol=1e-12)

    def test_standardization_fixes_linear_unit_changes(self):
        baseline = phiid_full(*self.data, measure="SX_GAUSSIAN")
        converted = self.data * np.array([3.0, 0.2, 8.0, 0.5])[:, None] + 5.0
        changed = phiid_full(*converted, measure="SX_GAUSSIAN")
        np.testing.assert_allclose(
            [baseline.atoms[name] for name in ATOM_NAMES],
            [changed.atoms[name] for name in ATOM_NAMES],
            atol=1e-12,
        )

    def test_swapping_past_and_future_transposes_atoms(self):
        forward = phiid_full(*self.data, measure="SX_GAUSSIAN")
        reverse = phiid_full(*self.data[[2, 3, 0, 1]], measure="SX_GAUSSIAN")
        matrix_forward = np.array([forward.atoms[name] for name in ATOM_NAMES]).reshape(4, 4)
        matrix_reverse = np.array([reverse.atoms[name] for name in ATOM_NAMES]).reshape(4, 4)
        np.testing.assert_allclose(matrix_forward, matrix_reverse.T, atol=1e-12)

    def test_rejects_singular_joint_law(self):
        a, b, _, d = self.data
        with self.assertRaisesRegex(ValueError, "nonsingular"):
            phiid_full(a, b, a, d, measure="SX_GAUSSIAN")

    def test_two_signal_lagged_api(self):
        signals = self.data[:2]
        result = phiid_full(signals, tau=2, measure="SX_GAUSSIAN", return_local=True)
        self.assertEqual(result.partition, ((0,), (1,)))
        self.assertEqual(result.local_atoms["sts"].shape, (signals.shape[1] - 2,))
        self.assertTrue(all(np.isfinite(list(result.atoms.values()))))

    def test_rejects_multivariate_group(self):
        with self.assertRaisesRegex(ValueError, "exactly two scalar"):
            phiid_full(self.data[:2], *self.data[1:], measure="SX_GAUSSIAN")


if __name__ == "__main__":
    unittest.main()
