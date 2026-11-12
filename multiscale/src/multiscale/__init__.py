"""Dynamical independence in the temporal and spectral domains."""

from .model import prepare_var, sfreqs, ss2trfun
from .objective import canonical_band_objective_and_grad, canonical_band_value, cak2ddx, cak2ddxgrad
from .optimise import canonical_opt_dds_mruns, opt_gd_ddx_mruns, run_two_stage, random_initial_bases
from .riccati import mdare, iss2dd
from .integrated import trfun2dd, trfun2ddgrad, opt_gd_dds_mruns
from .fit import tsdata_to_var, tsdata_to_varmo, fit_var_auto, VARFit, VARModelOrder, SelectedVAR

__all__ = ["prepare_var", "sfreqs", "ss2trfun", "canonical_band_objective_and_grad",
           "canonical_band_value",
           "cak2ddx", "cak2ddxgrad", "canonical_opt_dds_mruns", "opt_gd_ddx_mruns",
           "run_two_stage", "random_initial_bases", "mdare", "iss2dd",
           "trfun2dd", "trfun2ddgrad", "opt_gd_dds_mruns",
           "tsdata_to_var", "tsdata_to_varmo", "fit_var_auto", "VARFit",
           "VARModelOrder", "SelectedVAR"]
