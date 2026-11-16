"""Experimental two-sided Gaussian shared-exclusions PhiID.

Ehrlich et al. (2024) define continuous SX for a single target using the
quasi-density of a source disjunction (a sum of marginal densities). This
module extends that construction to both sides of Varley's PhiID lattice:

    i_sx(alpha -> beta) = log2[Q(alpha,beta) / (Q(alpha) Q(beta))]

where Q(alpha,beta) sums the joint densities for every pair of constituent
source and target events. This *two-sided* extension is a research proposal,
not an estimator established by either paper. It is defined here only for two
scalar sources and two scalar targets with a nonsingular joint Gaussian law.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .phiid import ATOM_NAMES, PHIID_LATTICE, PhiIDResult


_ANTICHAINS = (
    ((0,), (1,)),  # redundant
    ((0,),),        # first unique
    ((1,),),        # second unique
    ((0, 1),),      # synergistic
)
_REDUNDANCY_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (2, 0), (3, 0),
    (1, 1), (1, 2), (2, 1), (2, 2),
    (3, 1), (3, 2), (1, 3), (2, 3), (3, 3),
)


def phiid_gaussian_shared_exclusions(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    *,
    return_local: bool,
) -> PhiIDResult:
    """Estimate the proposed two-sided Gaussian SX decomposition in bits.

    Each input must describe one scalar variable. All four are standardized
    separately to fix the relative density scale, an explicit gauge choice.
    Near-singular fitted covariance matrices are rejected rather than ridged.
    """
    groups = tuple(np.asarray(v, dtype=float) for v in (x1, x2, y1, y2))
    if any(v.ndim != 2 or v.shape[0] != 1 for v in groups):
        raise ValueError("Gaussian SX requires exactly two scalar past and two scalar future variables.")
    data = np.vstack(groups)
    if not np.all(np.isfinite(data)):
        raise ValueError("Gaussian SX requires finite observations.")
    standard_deviation = np.std(data, axis=1, ddof=0)
    if np.any(standard_deviation <= 1e-12):
        raise ValueError("Gaussian SX requires nonconstant variables.")
    z = (data - np.mean(data, axis=1, keepdims=True)) / standard_deviation[:, None]
    covariance = np.cov(z)
    if np.linalg.eigvalsh(covariance)[0] <= 1e-10:
        raise ValueError("Gaussian SX requires a nonsingular joint covariance; add justified observation noise.")

    # Fifteen fitted marginal log-densities, shared by all sixteen lattice nodes.
    log_densities: dict[tuple[int, ...], np.ndarray] = {}
    for mask in range(1, 16):
        indices = tuple(i for i in range(4) if mask & (1 << i))
        values = multivariate_normal.logpdf(
            z[list(indices)].T,
            mean=np.zeros(len(indices)),
            cov=covariance[np.ix_(indices, indices)],
            allow_singular=False,
        )
        log_densities[indices] = np.atleast_1d(values)

    def log_quasi_density(components: tuple[tuple[int, ...], ...]) -> np.ndarray:
        return logsumexp(np.vstack([log_densities[group] for group in components]), axis=0)

    sources = _ANTICHAINS
    targets = tuple(
        tuple(tuple(i + 2 for i in group) for group in antichain)
        for antichain in _ANTICHAINS
    )
    source_log = [log_quasi_density(a) for a in sources]
    target_log = [log_quasi_density(a) for a in targets]
    redundancies = []
    for source_index, target_index in _REDUNDANCY_ORDER:
        joint = tuple(
            tuple(sorted(source_group + target_group))
            for source_group in sources[source_index]
            for target_group in targets[target_index]
        )
        local = (
            log_quasi_density(joint)
            - source_log[source_index]
            - target_log[target_index]
        ) / np.log(2.0)
        redundancies.append(local)

    partials = np.linalg.solve(PHIID_LATTICE, np.vstack(redundancies))
    local_atoms = {name: partials[i] for i, name in enumerate(ATOM_NAMES)}
    atoms = {name: float(np.mean(values)) for name, values in local_atoms.items()}
    return PhiIDResult(atoms=atoms, local_atoms=local_atoms if return_local else None)
