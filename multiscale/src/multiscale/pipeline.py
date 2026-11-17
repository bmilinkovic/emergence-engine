"""Proxy preoptimization followed by either actual DD calculation route."""

import numpy as np

from .integrated import opt_gd_dds_mruns
from .model import itransform_subspace, prepare_var, prepare_ss
from .optimise import opt_gd_ddx_mruns, cluster_representatives
from .temporal import opt_es_dd_mruns


def run_full_dd(a, v, fs, fres, initial_bases, *,
                final_method="spectral", pre_iters=10000,
                spectral_iters=1000, state_space_iters=100,
                cluster_tol=1e-10, spectral_sig=0.1,
                state_space_sig=1e-4, es_rule=0.2, seed=None,
                history=False, state_space_workers=1):
    """Run SSDI-1 proxy DDx and select either or both actual DD optimisers.

    ``spectral`` uses ``trfun2dd``/``trfun2ddgrad`` and gradient descent on
    the full transfer-function spectrum. ``state_space`` uses ``iss2dd`` and
    the original 1+1 evolution strategy on the state-space model. ``both``
    starts each from the same proxy representatives; ``spectral_then_state_space``
    follows the original simulation's sequential refinement path.

    This is full-spectrum DD. For frequency-band canonical DD use
    ``run_two_stage`` or ``run_grid_parallel`` instead.
    """
    return _run_prepared_dd(
        prepare_var(a, v, fs, fres), v, initial_bases,
        final_method=final_method, pre_iters=pre_iters,
        spectral_iters=spectral_iters, state_space_iters=state_space_iters,
        cluster_tol=cluster_tol, spectral_sig=spectral_sig,
        state_space_sig=state_space_sig, es_rule=es_rule, seed=seed,
        history=history, state_space_workers=state_space_workers)


def _run_prepared_dd(prepared, v, initial_bases, *,
                     final_method="spectral", pre_iters=10000,
                     spectral_iters=1000, state_space_iters=100,
                     cluster_tol=1e-10, spectral_sig=.1,
                     state_space_sig=1e-4, es_rule=.2, seed=None,
                     history=False, state_space_workers=1):
    methods = {"spectral", "state_space", "both", "spectral_then_state_space"}
    if final_method not in methods:
        raise ValueError(f"final_method must be one of {sorted(methods)}")
    ass, c, k, cak, h, freqs = prepared
    v, initial_bases = np.asarray(v, float), np.asarray(initial_bases, float)
    if initial_bases.ndim != 3 or initial_bases.shape[0] != c.shape[0]:
        raise ValueError("initial bases must have shape (regions,m,runs)")
    if not 1 <= initial_bases.shape[1] < c.shape[0]:
        raise ValueError("m must satisfy 1 <= m < regions")
    pre = opt_gd_ddx_mruns(cak, initial_bases, niters=pre_iters,
                            history=history)
    original_pre = itransform_subspace(pre.bases, v)
    reps, sizes = cluster_representatives(original_pre, cluster_tol)
    starts = pre.bases[:, :, reps]
    out = dict(pre=pre, pre_original_bases=original_pre,
               representatives=reps, cluster_sizes=sizes,
               frequencies=freqs)
    if final_method in ("spectral", "both", "spectral_then_state_space"):
        spectral = opt_gd_dds_mruns(h, starts, niters=spectral_iters,
                                    sig0=spectral_sig, tol=1e-10,
                                    history=history)
        out["spectral"] = spectral
        out["spectral_original_bases"] = itransform_subspace(spectral.bases, v)
    if final_method in ("state_space", "both", "spectral_then_state_space"):
        state_starts = (out["spectral"].bases if final_method ==
                        "spectral_then_state_space" else starts)
        state = opt_es_dd_mruns(ass, c, k, state_starts,
                                maxiters=state_space_iters,
                                sig=state_space_sig, rule=es_rule,
                                seed=seed, history=history,
                                workers=state_space_workers)
        out["state_space"] = state
        out["state_space_original_bases"] = itransform_subspace(state.bases, v)
    return out


def run_full_dd_ss(fit, fs, fres, initial_bases, **kwargs):
    """Run full-spectrum DD on a directly fitted SSFit innovations model.

    Accepts the same optimisation options as run_full_dd.
    """
    return _run_prepared_dd(prepare_ss(fit.A, fit.C, fit.K, fit.V, fs, fres),
                            fit.V, initial_bases, **kwargs)
