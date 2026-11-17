"""Direct CCA innovations-state-space estimation adapted from MVGC2 (GPL-3.0)."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, lstsq, solve_triangular
from .fit import fit_var_auto


@dataclass
class SSModelOrder:
    order: int
    maximum: int
    singular_values: np.ndarray
    svc_curve: np.ndarray
    horizons: tuple[int, int]


@dataclass
class SSFit:
    A: np.ndarray
    C: np.ndarray
    K: np.ndarray
    V: np.ndarray
    states: np.ndarray
    residuals: np.ndarray
    mean: np.ndarray
    selection: SSModelOrder
    order: int
    var_order: int | None = None

    @property
    def diagnostics(self):
        radius = lambda a: float(np.max(np.abs(np.linalg.eigvals(a))))
        return {"state_spectral_radius": radius(self.A),
                "predictor_spectral_radius": radius(self.A-self.K@self.C),
                "innovation_min_eigenvalue": float(np.linalg.eigvalsh(self.V).min())}


def _right_divide(y, x):
    return lstsq(x.T, y.T, lapack_driver="gelsy")[0].T


def _cca(data, pf):
    x = np.asarray(data, dtype=float)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("data must be finite (nodes, samples[, trials])")
    horizons = np.asarray(pf).reshape(-1)
    if (len(horizons) not in (1, 2) or not np.isfinite(horizons).all()
            or np.any(horizons < 1) or np.any(horizons != np.floor(horizons))):
        raise ValueError("pf must be a positive integer or (past, future) pair")
    p, f = (int(horizons[0]), int(horizons[-1]))
    n, m, trials = x.shape
    if p+f >= m:
        raise ValueError("past + future horizons must be smaller than samples per trial")
    mean = x.mean(axis=(1, 2), keepdims=True)
    x = x-mean
    mp, mh = m-p, m-p-f+1
    # Preserve MATLAB column ordering: node, lag, time, trial.
    future = np.stack([x[:, p+k:p+k+mh, :] for k in range(f)], axis=1)
    past = np.stack([x[:, p-1-k:m-k, :] for k in range(p)], axis=1)
    xf = future.reshape(n*f, mh*trials, order="F")
    xp = past[:, :, :mh, :].reshape(n*p, mh*trials, order="F")
    XP = past.reshape(n*p, (mp+1)*trials, order="F")
    try:
        wf = cholesky(xf@xf.T/(mh*trials), lower=True)
        wp = cholesky(xp@xp.T/(mh*trials), lower=True)
    except np.linalg.LinAlgError as error:
        raise ValueError("CCA horizon covariance is not positive definite; reduce horizons or check data rank") from error
    beta = _right_divide(xf, xp)
    _, singular, vh = np.linalg.svd(solve_triangular(wf, beta@wp, lower=True), full_matrices=False)
    maximum = n*min(p, f)
    # Preserve MVGC2's unsquared singular values and zero-based selected order.
    with np.errstate(divide="ignore", invalid="ignore"):
        svc = -np.log1p(-np.r_[singular[1:], 0.0]) + 2*n*np.arange(1, maximum+1)*np.log(mh*trials)/(mh*trials)
    selection = SSModelOrder(int(np.nanargmin(svc)), maximum, singular, svc, (p, f))
    return x, mean, XP, wp, vh, selection


def tsdata_to_ssmo(data, pf):
    """MVGC2 SVC selection, including its original indexing convention."""
    return _cca(data, pf)[-1]


def tsdata_to_ss(data, pf, r=None):
    """Fit an innovations model directly from time series using CCA.

    pf is a scalar horizon or (past, future). r is the latent state dimension;
    None selects it with MVGC2 SVC. No VAR is fitted in this function.
    States and residuals retain a trials axis, including for 2-D input.
    """
    x, mean, XP, wp, vh, selection = _cca(data, pf)
    r = selection.order if r is None else r
    if not np.isscalar(r) or not np.isfinite(r) or int(r) != r or not 1 <= r <= selection.maximum:
        raise ValueError("state order must be in 1..n*min(p,f); SVC may select zero (no retained dynamic state)")
    r = int(r)
    n, m, trials = x.shape
    p, _ = selection.horizons
    mp, count = m-p, trials*(m-p)
    projection = solve_triangular(wp.T, (np.sqrt(selection.singular_values[:r])[:, None]*vh[:r]).T, lower=False).T
    z = (projection@XP).reshape(r, mp+1, trials, order="F")
    current = z[:, :-1, :].reshape(r, count, order="F")
    following = z[:, 1:, :].reshape(r, count, order="F")
    observations = x[:, p:, :].reshape(n, count, order="F")
    c = _right_divide(observations, current)
    e = observations-c@current
    v = e@e.T/(count-1)
    ak = _right_divide(following, np.vstack((current, e)))
    return SSFit(ak[:, :r], c, ak[:, r:], v, z,
                 e.reshape(n, mp, trials, order="F"), mean[:, 0, 0], selection, r)


def fit_ss_auto(data, *, pf=None, r=None, pmax=20):
    """Direct CCA fit; optional VAR-AIC pilot chooses pf=2*VAR order.

    Supplying pf skips the pilot. SVC selects r unless explicitly supplied.
    An AIC order of zero requires explicit horizons rather than inventing one.
    """
    pilot = None
    if pf is None:
        pilot = fit_var_auto(data, pmax=pmax, criterion="aic").order
        if pilot == 0:
            raise ValueError("VAR-AIC selected zero; specify pf explicitly for a CCA fit")
        pf = 2*pilot
    result = tsdata_to_ss(data, pf, r)
    result.var_order = pilot
    return result
