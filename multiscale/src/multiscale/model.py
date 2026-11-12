"""The MVGC2/SSDI-1 VAR-to-innovations path used by DMT.

Array convention is (output, input, lag), as in MATLAB. All transforms use
the original Cholesky and SVD constructions, not covariance whitening by an
eigendecomposition or QR basis extraction.
"""

import numpy as np
from scipy.linalg import cholesky


def orthonormalise(x):
    return np.linalg.svd(np.asarray(x, dtype=float), full_matrices=False)[0]


def transform_var(a, v):
    a, v = np.asarray(a, float), np.asarray(v, float)
    root = cholesky(v, lower=True)
    inverse = np.linalg.inv(root)
    transformed = np.empty_like(a)
    for k in range(a.shape[2]):
        transformed[:, :, k] = inverse @ a[:, :, k] @ root
    return transformed


def itransform_subspace(bases, v):
    """SSDI-1 pullback through inverse transpose Cholesky, then SVD."""
    inverse = np.linalg.inv(cholesky(np.asarray(v, float), lower=True).T)
    bases = np.asarray(bases, float)
    if bases.ndim == 2:
        return orthonormalise(inverse @ bases)
    out = np.empty_like(bases)
    for i in range(bases.shape[2]):
        out[:, :, i] = orthonormalise(inverse @ bases[:, :, i])
    return out


def var_to_ss(a):
    a = np.asarray(a, float)
    n, n2, p = a.shape
    if n != n2 or p < 1:
        raise ValueError("VAR coefficients must have shape (n,n,p) with p >= 1")
    c = np.concatenate([a[:, :, k] for k in range(p)], axis=1)
    ass = np.zeros((n*p, n*p))
    ass[:n, :] = c
    ass[n:, :n*(p-1)] = np.eye(n*(p-1))
    k = np.zeros((n*p, n))
    k[:n, :] = np.eye(n)
    return ass, c, k


def iss2cak(a, c, k):
    n, r = c.shape
    cak = np.empty((n, n, r))
    ak = np.eye(r)
    for lag in range(r):
        cak[:, :, lag] = c @ ak @ k
        ak = ak @ a
    return cak


def sfreqs(fres, fs):
    return np.arange(fres + 1, dtype=float) * (fs / (2*fres))


def ss2trfun(a, c, k, fres):
    """MVGC2 ss2trfun: I + C(zI-A)^(-1)K at z=exp(i*pi*j/fres)."""
    n, r = c.shape
    h = np.empty((n, n, fres+1), dtype=complex)
    ir, inn = np.eye(r), np.eye(n)
    for j in range(fres+1):
        z = np.exp(1j*np.pi*j/fres)
        h[:, :, j] = inn + c @ np.linalg.solve(z*ir-a, k)
    return h


def prepare_var(a, v, fs, fres):
    """Return transformed model, CAK proxy, transfer function, frequencies."""
    transformed = transform_var(a, v)
    ass, c, k = var_to_ss(transformed)
    return ass, c, k, iss2cak(ass, c, k), ss2trfun(ass, c, k, fres), sfreqs(fres, fs)
