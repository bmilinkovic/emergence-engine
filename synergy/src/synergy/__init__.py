"""Phi_ID computation and optional analysis for arbitrary node sets."""

__version__ = "0.5.0"

from .phiid import ATOM_NAMES, PhiIDResult, phiid_full
from .phiid_discrete import phiid_full_discrete
from .partition import minimum_information_partition
from .pairwise import pairwise_atoms

__all__ = ["ATOM_NAMES", "PhiIDResult", "phiid_full", "phiid_full_discrete", "minimum_information_partition", "pairwise_atoms"]
