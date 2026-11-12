"""Batch pairwise Gaussian MMI/CCS for arbitrary channel or region sets.

Fits the same standardised Gaussian and 1e-10 covariance ridge. Gaussian local
MI is a quadratic polynomial; evaluate it with Numba rather than repeatedly
fitting distributions. CCS still evaluates EVERY lagged sample, without binning,
decimation, or covariance-only approximation. Outputs are in natural-log nats.
"""
import numpy as np
try:
    from numba import njit, prange
except ImportError:  # The NumPy implementation remains available on clean installs.
    def njit(*args, **kwargs):
        return lambda function: function
    prange = range
    _HAS_NUMBA = False
else:
    _HAS_NUMBA = True
from .phiid import ATOM_NAMES, PHIID_LATTICE

_LATTICE_INV = np.linalg.inv(PHIID_LATTICE)

# Ordering of Gaussian MIs used in phiid._four_vector_phiid.
MI_SETS = [((0,), (2,)), ((0,), (3,)), ((1,), (2,)), ((1,), (3,)),
           ((0, 1), (2,)), ((0, 1), (3,)), ((0,), (2, 3)),
           ((1,), (2, 3)), ((0, 1), (2, 3))]
TRI = np.triu_indices(4)
STS_WEIGHTS = _LATTICE_INV[15]


def coefficients(cov):
    """Return 9 local-MI polynomials [constant, ten upper quadratic terms]."""
    # cov: batch x 4 x 4, sample covariance after population-SD normalisation.
    entropy = {}
    for mask in range(1, 16):
        idx = [k for k in range(4) if mask & (1 << k)]
        c = cov[:, idx, :][:, :, idx] + 1e-10 * np.eye(len(idx))
        sign, logdet = np.linalg.slogdet(c)
        if np.any(sign <= 0):
            raise ValueError('Non-positive Gaussian covariance')
        inv = np.linalg.inv(c)
        q = np.zeros_like(cov)
        for i, a in enumerate(idx):
            for j, b in enumerate(idx):
                q[:, a, b] = .5 * inv[:, i, j]
        terms = q[:, TRI[0], TRI[1]].copy()
        terms[:, TRI[0] != TRI[1]] *= 2
        entropy[mask] = np.column_stack((.5 * (len(idx) * np.log(2*np.pi) + logdet), terms))
    out = []
    for a, b in MI_SETS:
        ma = sum(1 << k for k in a)
        mb = sum(1 << k for k in b)
        out.append(entropy[ma] + entropy[mb] - entropy[ma | mb])
    return np.stack(out, axis=1)


@njit
def ccs(a, b, joint):
    co = a + b - joint
    sign = np.sign(a)
    return co if sign == np.sign(b) and sign == np.sign(joint) and sign == np.sign(co) else 0.


@njit(parallel=True)
def evaluate_ccs(data, tau, pairs, means, stds, coefs, weights):
    n = data.shape[1] - tau
    out = np.zeros((len(pairs), 2))
    for p in prange(len(pairs)):
        i, j = pairs[p]
        coeff = coefs[p]
        mean = np.array([means[0, i], means[0, j], means[1, i], means[1, j]])
        scale = np.array([stds[0, i], stds[0, j], stds[1, i], stds[1, j]])
        v = np.empty(4)
        terms = np.empty(11)
        mi = np.empty(9)
        reds = np.empty(16)
        rtr_sum = 0.
        sts_sum = 0.
        for t in range(n):
            v[0] = (data[i,t] - mean[0]) / scale[0]
            v[1] = (data[j,t] - mean[1]) / scale[1]
            v[2] = (data[i,t+tau] - mean[2]) / scale[2]
            v[3] = (data[j,t+tau] - mean[3]) / scale[3]
            terms[0] = 1.
            k = 1
            for a in range(4):
                for b in range(a,4):
                    terms[k] = v[a]*v[b]
                    k += 1
            for a in range(9):
                total = 0.
                for b in range(11):
                    total += coeff[a,b] * terms[b]
                mi[a] = total
            a,b,c,d,e,f,g,h,total = mi
            ra,rb,rc = ccs(a,c,e),ccs(b,d,f),ccs(g,h,total)
            rd,re,rf = ccs(a,b,g),ccs(c,d,h),ccs(e,f,total)
            co = -a-b-c-d+g+h+e+f-total+ra+rb-rc+rd+re-rf
            sg = np.sign(a)
            rtr = co if sg == np.sign(b) and sg == np.sign(c) and sg == np.sign(d) and sg == np.sign(co) else 0.
            reds[0],reds[1],reds[2],reds[3] = rtr,ra,rb,rc
            reds[4],reds[5],reds[6] = rd,re,rf
            reds[7],reds[8],reds[9],reds[10] = a,b,c,d
            reds[11],reds[12],reds[13],reds[14],reds[15] = e,f,g,h,total
            sts = 0.
            for k in range(16):
                sts += weights[k] * reds[k]
            rtr_sum += rtr
            sts_sum += sts
        out[p,0] = rtr_sum/n
        out[p,1] = sts_sum/n
    return out


if not _HAS_NUMBA:
    def evaluate_ccs(data, tau, pairs, means, stds, coefs, weights):
        """Vectorized fallback; processes one bounded pair batch at a time."""
        n = data.shape[1] - tau
        source = np.stack((data[pairs[:, 0], :n], data[pairs[:, 1], :n],
                           data[pairs[:, 0], tau:], data[pairs[:, 1], tau:]), axis=2)
        centre = np.stack((means[0, pairs[:, 0]], means[0, pairs[:, 1]],
                           means[1, pairs[:, 0]], means[1, pairs[:, 1]]), axis=1)
        scale = np.stack((stds[0, pairs[:, 0]], stds[0, pairs[:, 1]],
                          stds[1, pairs[:, 0]], stds[1, pairs[:, 1]]), axis=1)
        v = (source - centre[:, None, :]) / scale[:, None, :]
        terms = np.concatenate((np.ones((*v.shape[:2], 1)),
                                v[:, :, TRI[0]] * v[:, :, TRI[1]]), axis=2)
        mi = np.einsum('pik,ptk->pti', coefs, terms)
        a,b,c,d,e,f,g,h,total = np.moveaxis(mi, 2, 0)
        def common(first, second, joint):
            co = first + second - joint
            sign = np.sign(first)
            return np.where((sign == np.sign(second)) &
                            (sign == np.sign(joint)) &
                            (sign == np.sign(co)), co, 0.)
        ra,rb,rc = common(a,c,e),common(b,d,f),common(g,h,total)
        rd,re,rf = common(a,b,g),common(c,d,h),common(e,f,total)
        double = -a-b-c-d+g+h+e+f-total+ra+rb-rc+rd+re-rf
        sign = np.sign(a)
        rtr = np.where((sign == np.sign(b)) & (sign == np.sign(c)) &
                       (sign == np.sign(d)) & (sign == np.sign(double)), double, 0.)
        reds = np.stack((rtr,ra,rb,rc,rd,re,rf,a,b,c,d,e,f,g,h,total), axis=2)
        sts = np.einsum('ptk,k->pt', reds, weights)
        return np.stack((rtr.mean(axis=1), sts.mean(axis=1)), axis=1)


@njit(parallel=True)
def evaluate_ccs_reds(data, tau, pairs, means, stds, coefs):
    """Mean of the seven CCS-specific redundancy terms for every pair."""
    n = data.shape[1] - tau
    out = np.zeros((len(pairs), 7))
    for p in prange(len(pairs)):
        i, j = pairs[p]
        coeff = coefs[p]
        mean = np.array([means[0, i], means[0, j], means[1, i], means[1, j]])
        scale = np.array([stds[0, i], stds[0, j], stds[1, i], stds[1, j]])
        terms = np.empty(11)
        mi = np.empty(9)
        for t in range(n):
            v0 = (data[i, t] - mean[0]) / scale[0]
            v1 = (data[j, t] - mean[1]) / scale[1]
            v2 = (data[i, t + tau] - mean[2]) / scale[2]
            v3 = (data[j, t + tau] - mean[3]) / scale[3]
            v = np.array([v0, v1, v2, v3])
            terms[0] = 1.
            k = 1
            for aa in range(4):
                for bb in range(aa, 4):
                    terms[k] = v[aa] * v[bb]
                    k += 1
            for aa in range(9):
                value = 0.
                for bb in range(11):
                    value += coeff[aa, bb] * terms[bb]
                mi[aa] = value
            a, b, c, d, e, f, g, h, total = mi
            ra, rb, rc = ccs(a, c, e), ccs(b, d, f), ccs(g, h, total)
            rd, re, rf = ccs(a, b, g), ccs(c, d, h), ccs(e, f, total)
            co = -a-b-c-d+g+h+e+f-total+ra+rb-rc+rd+re-rf
            sg = np.sign(a)
            rtr = co if sg == np.sign(b) and sg == np.sign(c) and sg == np.sign(d) and sg == np.sign(co) else 0.
            out[p, 0] += rtr
            out[p, 1] += ra
            out[p, 2] += rb
            out[p, 3] += rc
            out[p, 4] += rd
            out[p, 5] += re
            out[p, 6] += rf
        out[p] /= n
    return out


if not _HAS_NUMBA:
    def evaluate_ccs_reds(data, tau, pairs, means, stds, coefs):
        n = data.shape[1] - tau
        source = np.stack((data[pairs[:, 0], :n], data[pairs[:, 1], :n],
                           data[pairs[:, 0], tau:], data[pairs[:, 1], tau:]), axis=2)
        centre = np.stack((means[0, pairs[:, 0]], means[0, pairs[:, 1]],
                           means[1, pairs[:, 0]], means[1, pairs[:, 1]]), axis=1)
        scale = np.stack((stds[0, pairs[:, 0]], stds[0, pairs[:, 1]],
                          stds[1, pairs[:, 0]], stds[1, pairs[:, 1]]), axis=1)
        v = (source - centre[:, None, :]) / scale[:, None, :]
        terms = np.concatenate((np.ones((*v.shape[:2], 1)),
                                v[:, :, TRI[0]] * v[:, :, TRI[1]]), axis=2)
        mi = np.einsum('pik,ptk->pti', coefs, terms)
        a,b,c,d,e,f,g,h,total = np.moveaxis(mi, 2, 0)
        def common(first, second, joint):
            co = first + second - joint
            sign = np.sign(first)
            return np.where((sign == np.sign(second)) &
                            (sign == np.sign(joint)) &
                            (sign == np.sign(co)), co, 0.)
        ra,rb,rc = common(a,c,e),common(b,d,f),common(g,h,total)
        rd,re,rf = common(a,b,g),common(c,d,h),common(e,f,total)
        double = -a-b-c-d+g+h+e+f-total+ra+rb-rc+rd+re-rf
        sign = np.sign(a)
        rtr = np.where((sign == np.sign(b)) & (sign == np.sign(c)) &
                       (sign == np.sign(d)) & (sign == np.sign(double)), double, 0.)
        return np.stack((rtr,ra,rb,rc,rd,re,rf), axis=2).mean(axis=1)


def pairwise_atoms(data, tau=1, methods=('mmi', 'ccs'), progress=None,
                   all_atoms=False):
    """Compute Gaussian pairwise Phi_ID matrices.

    With ``all_atoms=True``, return all 16 ordered atoms per estimator.
    Entry ``[i,j]`` uses node ``i`` as the X source and node ``j`` as Y;
    opposite directed entries swap X/Y atom labels as in MATLAB PhiIDFull.
    The default returns only symmetric RTR and STS matrices.
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    if data.ndim != 2 or not isinstance(tau, int) or not 1 <= tau < data.shape[1]-4:
        raise ValueError('Require channels x samples and valid integer tau')
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite signal; no silent channel/pair exclusions')
    if not set(methods) <= {'mmi','ccs'}:
        raise ValueError('Supported methods: mmi, ccs')
    d, t = data.shape
    n = t-tau
    means = np.stack([data[:,:n].mean(axis=1),data[:,tau:].mean(axis=1)])
    stds = np.stack([data[:,:n].std(axis=1),data[:,tau:].std(axis=1)])
    if np.any(stds < 1e-12):
        raise ValueError('Constant/near-constant channel')
    # Accumulate centred cross products in chunks to bound working memory.
    scatter = np.zeros((2*d,2*d))
    for start in range(0,n,32768):
        stop = min(n,start+32768)
        block = np.concatenate([data[:,start:stop],data[:,start+tau:stop+tau]])
        block -= means.reshape(-1,1)
        block /= stds.reshape(-1,1)
        scatter += block @ block.T
    cov = scatter/(n-1)
    pairs = np.array(list(zip(*np.triu_indices(d,1))))
    indices = np.column_stack([pairs,pairs+d])
    paircov = cov[indices[:,:,None],indices[:,None,:]]
    coefs = coefficients(paircov)
    # Exact sample mean of quadratic local MI, including finite-N/ridge terms.
    basis = paircov[:,TRI[0],TRI[1]]*(n-1)/n
    means_mi = coefs[:,:,0] + np.einsum('pij,pj->pi',coefs[:,:,1:],basis)
    results = {}
    if 'mmi' in methods:
        a,b,c,e2,e,f,g,h,total = means_mi.T
        reds = np.array([np.min(means_mi[:,:4],axis=1),np.minimum(a,c),np.minimum(b,e2),
                         np.minimum(g,h),np.minimum(a,b),np.minimum(c,e2),np.minimum(e,f),
                         a,b,c,e2,e,f,g,h,total])
        atoms = _LATTICE_INV @ reds
        results['mmi'] = atoms if all_atoms else (atoms[0],atoms[15])
    if 'ccs' in methods:
        chunks = []
        for start in range(0,len(pairs),256):
            end = min(start+256,len(pairs))
            if all_atoms:
                chunks.append(evaluate_ccs_reds(data,tau,pairs[start:end],means,stds,coefs[start:end]))
            else:
                chunks.append(evaluate_ccs(data,tau,pairs[start:end],means,stds,coefs[start:end],STS_WEIGHTS))
            if progress:
                progress(end,len(pairs))
        values = np.concatenate(chunks)
        if all_atoms:
            red = np.vstack((values.T, means_mi.T))
            results['ccs'] = _LATTICE_INV @ red
        else:
            results['ccs'] = (values[:,0],values[:,1])
    matrices = {}
    swap = {name: name.replace('x', '#').replace('y', 'x').replace('#', 'y')
            for name in ATOM_NAMES}
    for method,values_by_atom in results.items():
        matrices[method] = {}
        items = zip(ATOM_NAMES, values_by_atom) if all_atoms else zip(('rtr','sts'), values_by_atom)
        for atom,values in items:
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite {method}/{atom}')
            mat = np.zeros((d,d))
            mat[pairs[:,0],pairs[:,1]] = values
            matrices[method][atom] = mat
        for atom, mat in matrices[method].items():
            reverse = matrices[method][swap[atom]]
            mat[pairs[:,1],pairs[:,0]] = reverse[pairs[:,0],pairs[:,1]]
    return matrices
