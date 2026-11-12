"""MVGC2's extended-pencil QZ discrete Riccati solver.

This follows ``mdare.m``: build H/J, deflate by QR, order the *outside*
unit-circle generalized eigenvalues, then recover the invariant subspace.
It deliberately does not call a generic discrete-ARE wrapper.
"""

import numpy as np
from scipy.linalg import lu, ordqz, qr


def mdare(a, c, q, r=None, s=None):
    a, c, q = np.asarray(a, float), np.asarray(c, float), np.asarray(q, float)
    dim = a.shape[0]
    obs = c.shape[0]
    r = np.eye(obs) if r is None else np.asarray(r, float)
    s = np.zeros((dim, obs)) if s is None else np.asarray(s, float)
    if a.shape != (dim, dim) or c.shape != (obs, dim) or q.shape != a.shape:
        raise ValueError("invalid DARE dimensions")
    h = np.block([[a.T, np.zeros((dim, dim)), c.T],
                  [-q, np.eye(dim), -s],
                  [s.T, np.zeros((obs, dim)), r]])
    j = np.block([[np.eye(dim), np.zeros((dim, dim+obs))],
                  [np.zeros((dim, dim)), a, np.zeros((dim, obs))],
                  [np.zeros((obs, dim)), -c, np.zeros((obs, obs))]])
    # MATLAB qr(...), then q(:,n+1:end)' times the pencil.
    basis, _ = qr(h[:, 2*dim:], mode="full")
    projected_h = basis[:, obs:].T @ h[:, :2*dim]
    projected_j = basis[:, obs:].T @ j[:, :2*dim]
    jj, hh, alpha, beta, _, z = ordqz(projected_j, projected_h,
                                        sort="ouc", output="real")
    eigen = alpha/beta
    outside = np.abs(eigen) > 1
    if not np.all(outside[:dim]) or np.any(outside[dim:]):
        return None, None, -1.0, None
    p1, p2 = z[:dim, :dim], z[dim:, :dim]
    if np.linalg.cond(p1) > 1/np.finfo(float).eps:
        return None, None, -2.0, None
    p = np.linalg.solve(p1.T, p2.T).T
    p = (p+p.T)/2
    u = a @ p @ c.T+s
    v = c @ p @ c.T+r
    k = np.linalg.solve(v.T, u.T).T
    apa = a @ p @ a.T-p
    uk = u @ k.T
    rep = np.linalg.norm(apa-uk+q, 1)/(1+np.linalg.norm(apa, 1)+
                                       np.linalg.norm(uk, 1)+np.linalg.norm(q, 1))
    return k, v, float(rep), p


def iss2dd(l, a, c, k):
    """Temporal DI for an innovations SS with identity covariance."""
    q = k @ k.T
    _, v, rep, _ = mdare(a, l.T @ c, q, np.eye(l.shape[1]), k @ l)
    if rep < 0 or rep > 1e-8:
        return float("nan")
    sign, value = np.linalg.slogdet(v)
    return float(value) if sign > 0 else float("nan")
