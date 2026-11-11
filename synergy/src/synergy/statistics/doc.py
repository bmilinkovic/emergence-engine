"""DoC permutation and FDR methods copied from the original DoC analysis."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
COHORTS=("control","emcs","mcs","uws","coma")
SYSTEMS=("Sensorimotor","Frontal association","Orbitofrontal","Insula","Limbic","Parietal","Temporal","Occipital-ventral","Subcortical")

def perm_diff(x,y,nperm=20000,seed=16):
    x=np.asarray(x,float);y=np.asarray(y,float)
    x=x[np.isfinite(x)];y=y[np.isfinite(y)]
    obs=float(y.mean()-x.mean())
    pool=np.r_[x,y];n=len(x)
    rng=np.random.default_rng(seed)
    count=0
    for _ in range(nperm):
        z=rng.permutation(pool)
        count+=abs(z[n:].mean()-z[:n].mean())>=abs(obs)-1e-14
    p=(count+1)/(nperm+1)
    boots=[]
    for _ in range(4000):
        boots.append(rng.choice(y,len(y),replace=True).mean()-rng.choice(x,len(x),replace=True).mean())
    lo,hi=np.percentile(boots,[2.5,97.5])
    return obs,p,float(lo),float(hi)


def bh(p):
    p=np.asarray(p,float);q=np.full_like(p,np.nan)
    keep=np.isfinite(p);v=p[keep]
    if len(v):
        order=np.argsort(v);r=v[order]*len(v)/np.arange(1,len(v)+1)
        r=np.minimum.accumulate(r[::-1])[::-1]
        qv=np.empty(len(v));qv[order]=np.clip(r,0,1);q[keep]=qv
    return q


def cohort_pairwise_tests(macro_df):
    rows=[]
    for method in ("mmi","ccs"):
        for left,right in itertools.combinations(COHORTS,2):
            for system in SYSTEMS:
                a=macro_df[(macro_df.method==method)&(macro_df.cohort==left)&
                           (macro_df.macro_system==system)].macro_gradient_mean.to_numpy(float)
                b=macro_df[(macro_df.method==method)&(macro_df.cohort==right)&
                           (macro_df.macro_system==system)].macro_gradient_mean.to_numpy(float)
                # The bootstrap is also kept for small-group uncertainty.
                diff,p,lo,hi=perm_diff(a,b,nperm=10000,seed=6200+len(rows))
                rows.append(dict(method=method,cohort_a=left,cohort_b=right,macro_system=system,
                                 n_a=len(a),n_b=len(b),b_minus_a=diff,perm_p=p,
                                 bootstrap_95_low=lo,bootstrap_95_high=hi))
    out=pd.DataFrame(rows)
    out["q_fdr_90_per_method"]=out.groupby("method").perm_p.transform(lambda p:bh(p.to_numpy()))
    return out
