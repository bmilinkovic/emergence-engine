"""Discrete pointwise I_tau-sx double redundancy (Varley, 2023).

For source and target antichains, the local redundancy is
log2[P(source event AND target event) / (P(source event) P(target event))].
Each antichain event is the union of its observed component states.  Counts
are computed by inclusion-exclusion over empirical state frequencies.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .phiid import ATOM_NAMES, PHIID_LATTICE, PhiIDResult


_ANTICHAINS = (
    ((0,), (1,)),  # redundant
    ((0,),),        # first unique
    ((1,),),        # second unique
    ((0, 1),),      # synergistic
)


def _event_terms(antichain: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    """Return inclusion-exclusion coefficients keyed by variable bitmask."""
    terms: dict[int, int] = {}
    for size in range(1, len(antichain) + 1):
        for selected in combinations(antichain, size):
            mask = 0
            for component in selected:
                for variable in component:
                    mask |= 1 << variable
            terms[mask] = terms.get(mask, 0) + (-1 if size % 2 == 0 else 1)
    return terms


def _matching_counts(states: np.ndarray, mask: int) -> np.ndarray:
    """Count samples matching each observed state on the selected variables."""
    columns = [i for i in range(4) if mask & (1 << i)]
    _, inverse, counts = np.unique(
        states[columns].T, axis=0, return_inverse=True, return_counts=True
    )
    return counts[inverse]


def phiid_shared_exclusions(states: np.ndarray, return_local: bool) -> PhiIDResult:
    """Compute Varley's I_tau-sx PhiID for four integer-coded variables.

    ``states`` has rows [past 1, past 2, future 1, future 2]. Results are in
    bits. No pseudocount or smoothing is applied; all observed events have
    positive empirical probability.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] != 4 or states.shape[1] < 2:
        raise ValueError("states must have shape (4, samples), with at least two samples.")
    n = states.shape[1]
    counts = {mask: _matching_counts(states, mask) for mask in range(1, 16)}

    def count(terms: dict[int, int]) -> np.ndarray:
        result = np.zeros(n, dtype=np.int64)
        for mask, coefficient in terms.items():
            result += coefficient * counts[mask]
        if np.any(result <= 0):
            raise RuntimeError("Observed shared-exclusion event has zero count.")
        return result

    source_terms = [_event_terms(a) for a in _ANTICHAINS]
    target_terms = [
        _event_terms(tuple(tuple(i + 2 for i in component) for component in a))
        for a in _ANTICHAINS
    ]
    source_counts = [count(terms) for terms in source_terms]
    target_counts = [count(terms) for terms in target_terms]

    redundancies = []
    for source_index, target_index in (
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (2, 0), (3, 0),
        (1, 1), (1, 2), (2, 1), (2, 2),
        (3, 1), (3, 2), (1, 3), (2, 3), (3, 3),
    ):
        joint_terms: dict[int, int] = {}
        for source_mask, source_coefficient in source_terms[source_index].items():
            for target_mask, target_coefficient in target_terms[target_index].items():
                mask = source_mask | target_mask
                joint_terms[mask] = (
                    joint_terms.get(mask, 0) + source_coefficient * target_coefficient
                )
        joint_count = count(joint_terms)
        local = np.log2(
            (joint_count.astype(float) * n)
            / (
                source_counts[source_index].astype(float)
                * target_counts[target_index]
            )
        )
        redundancies.append(local)

    partials = np.linalg.solve(PHIID_LATTICE, np.vstack(redundancies))
    local_atoms = {name: partials[i] for i, name in enumerate(ATOM_NAMES)}
    atoms = {name: float(np.mean(values)) for name, values in local_atoms.items()}
    return PhiIDResult(atoms=atoms, local_atoms=local_atoms if return_local else None)
