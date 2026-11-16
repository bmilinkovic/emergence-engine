"""Independent checks for Varley's discrete shared-exclusions PhiID."""

import unittest
from itertools import product

import numpy as np

from synergy import ATOM_NAMES, phiid_full_discrete
from synergy.phiid import PHIID_LATTICE


ANTICHAINS = (
    ((0,), (1,)),
    ((0,),),
    ((1,),),
    ((0, 1),),
)
REDUNDANCY_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (2, 0), (3, 0),
    (1, 1), (1, 2), (2, 1), (2, 2),
    (3, 1), (3, 2), (1, 3), (2, 3), (3, 3),
)


def direct_event(states, observed, antichain):
    return np.logical_or.reduce([
        np.all(states[np.array(group)] == observed[np.array(group), None], axis=0)
        for group in antichain
    ])


class SharedExclusionsTests(unittest.TestCase):
    def test_matches_direct_probability_events_and_local_mi(self):
        states = np.array(list(product((0, 1), repeat=4)) +
                          [(0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1)]).T
        result = phiid_full_discrete(*states, measure="SX", return_local=True)
        atoms = np.vstack([result.local_atoms[name] for name in ATOM_NAMES])
        recovered_redundancies = PHIID_LATTICE @ atoms

        for sample, observed in enumerate(states.T):
            for row, (source_index, target_index) in enumerate(REDUNDANCY_ORDER):
                source = direct_event(states, observed, ANTICHAINS[source_index])
                target_groups = tuple(
                    tuple(i + 2 for i in group) for group in ANTICHAINS[target_index]
                )
                target = direct_event(states, observed, target_groups)
                expected = np.log2(
                    np.mean(source & target) / (np.mean(source) * np.mean(target))
                )
                self.assertAlmostEqual(recovered_redundancies[row, sample], expected)
            self.assertAlmostEqual(atoms[:, sample].sum(), recovered_redundancies[-1, sample])

    def test_perfect_copy_is_one_bit_redundancy(self):
        bit = np.array([0, 0, 1, 1] * 10)
        result = phiid_full_discrete(bit, bit, bit, bit, measure="SX")
        self.assertAlmostEqual(result.atoms["rtr"], 1.0)
        for name in ATOM_NAMES[1:]:
            self.assertAlmostEqual(result.atoms[name], 0.0)

    def test_varley_disintegrated_identity_table(self):
        """The two independent self-copying bits reproduce Table I (rounded)."""
        first = np.array([0, 0, 1, 1] * 10)
        second = np.array([0, 1, 0, 1] * 10)
        result = phiid_full_discrete(first, second, first, second, measure="SX")
        self.assertAlmostEqual(result.atoms["rtr"], 0.4150374992788438)
        self.assertAlmostEqual(result.atoms["xtx"], 0.5849625007211562)
        self.assertAlmostEqual(result.atoms["yty"], 0.5849625007211562)
        self.assertAlmostEqual(result.atoms["xty"], -0.4150374992788438)
        self.assertAlmostEqual(result.atoms["ytx"], -0.4150374992788438)
        self.assertAlmostEqual(sum(result.atoms.values()), 2.0)

    def test_continuous_input_requires_explicit_discretization(self):
        data = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]])
        with self.assertRaisesRegex(ValueError, "discretize continuous data"):
            phiid_full_discrete(data, measure="SX")


if __name__ == "__main__":
    unittest.main()
