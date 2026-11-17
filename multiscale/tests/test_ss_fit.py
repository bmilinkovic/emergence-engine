"""Numerical parity against independently executed MVGC2 MATLAB sources."""
from pathlib import Path
import numpy as np
import pytest
from scipy.io import loadmat
from multiscale import (tsdata_to_ss, tsdata_to_ssmo, fit_ss_auto,
                        prepare_ss, random_initial_bases, run_full_dd_ss)
from multiscale.model import ss2trfun

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize('name,pf', [('ss_mvgc2.mat', (4, 3)), ('ss_mvgc2_single.mat', 4)])
def test_reference_parity(name, pf):
    ref = loadmat(FIXTURES / name)
    fit = tsdata_to_ss(ref['X'], pf, 2)
    np.testing.assert_allclose(fit.V, ref['V'], atol=1e-12)
    expected_e = ref['E'] if ref['E'].ndim == 3 else ref['E'][:, :, None]
    np.testing.assert_allclose(fit.residuals, expected_e, atol=1e-12)
    np.testing.assert_allclose(ss2trfun(fit.A, fit.C, fit.K, 64),
                               ss2trfun(ref['A'], ref['C'], ref['K'], 64), atol=1e-11)
    # Latent coordinates are only identifiable up to an invertible transform.
    actual_z = fit.states.reshape(2, -1, order='F')
    reference_z = ref['Z'].reshape(2, -1, order='F')
    transform = np.linalg.lstsq(reference_z.T, actual_z.T, rcond=None)[0].T
    np.testing.assert_allclose(actual_z, transform@reference_z, atol=1e-11)
    np.testing.assert_allclose(fit.A@transform, transform@ref['A'], atol=1e-11)
    np.testing.assert_allclose(fit.C@transform, ref['C'], atol=1e-11)
    np.testing.assert_allclose(fit.K, transform@ref['K'], atol=1e-11)
    assert tsdata_to_ssmo(ref['X'], pf).order == int(ref['mosvc'].item())
    automatic = tsdata_to_ss(ref['X'], pf)
    assert automatic.order == int(ref['mosvc'].item())


def test_dd_adapter_and_pilot():
    x = loadmat(FIXTURES / 'ss_mvgc2.mat')['X']
    fit = fit_ss_auto(x, pf=(4, 3), r=2)
    assert fit.var_order is None
    a, c, k, _, h, _ = prepare_ss(fit.A, fit.C, fit.K, fit.V, 64, 16)
    root = np.linalg.cholesky(fit.V)
    original = ss2trfun(fit.A, fit.C, fit.K, 16)
    for i in range(h.shape[2]):
        np.testing.assert_allclose(root@h[:, :, i], original[:, :, i]@root, atol=1e-12)
    result = run_full_dd_ss(fit, 64, 16, random_initial_bases(3, 1, 2, seed=2),
                            final_method='both', pre_iters=2, spectral_iters=2,
                            state_space_iters=2, seed=2)
    assert np.isfinite(result['spectral'].values).all()
    assert np.isfinite(result['state_space'].values).all()
    automatic = fit_ss_auto(x, pmax=4, r=2)
    assert automatic.selection.horizons == (2*automatic.var_order,)*2


@pytest.mark.parametrize('pf', [0, -1, 1.5, (4, 600), (1, 2, 3)])
def test_invalid_horizons(pf):
    with pytest.raises(ValueError):
        tsdata_to_ss(np.random.default_rng(1).normal(size=(3, 100)), pf, 2)


def test_rank_and_order_errors():
    with pytest.raises(ValueError, match='positive definite'):
        tsdata_to_ss(np.ones((3, 100)), 3, 2)
    with pytest.raises(ValueError, match='state order'):
        tsdata_to_ss(np.random.default_rng(2).normal(size=(3, 100)), 3, 0)
