"""Masked DoC downstream analysis for repaired PhiID matrices."""
from __future__ import annotations
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from ..statistics.doc import perm_diff,bh,cohort_pairwise_tests
from ..statistics.pairwise import _subject_similarity_tests,_subject_graph_tests,_surface_gradient_roi_tests
from ..statistics.gradient import _perm_test_one_sample,_macro_system_from_base_label
from .metrics import threshold_top_density,weighted_global_efficiency,weighted_modularity
from ..plotting.gradient_violin import render_violin_grid
from ..plotting.gradient_surface import render_surface_panel
from ..plotting.pairwise_panels import _plot_similarity_paired,_plot_graph_paired
COHORTS=("control","emcs","mcs","uws","coma")
SYSTEMS=("Sensorimotor","Frontal association","Orbitofrontal","Insula","Limbic","Parietal","Temporal","Occipital-ventral","Subcortical")
U=np.triu_indices(90,1)

def rho(a,b,mask=None):
    x=np.asarray(a)[U];y=np.asarray(b)[U]
    good=np.isfinite(x)&np.isfinite(y)
    if mask is not None:good&=np.asarray(mask)[U]
    if good.sum()<3 or np.std(x[good])<1e-12 or np.std(y[good])<1e-12:return np.nan
    return float(stats.spearmanr(x[good],y[good]).statistic)


def grad(sts,rtr):
    valid=np.isfinite(np.diag(sts))&np.isfinite(np.diag(rtr))
    a,b=sts[np.ix_(valid,valid)],rtr[np.ix_(valid,valid)]
    out=np.full(90,np.nan)
    if valid.sum()>1:
        out[valid]=np.floor(stats.rankdata(a.mean(0))) - np.floor(stats.rankdata(b.mean(0)))
    return out


def run(input_dir: Path, output_dir: Path, manifest_path: Path, *, render: bool = True):
    """Recompute all subject summaries and inference from repaired PhiID matrices."""
    out=Path(output_dir)
    source=Path(input_dir)
    out.mkdir(parents=True,exist_ok=True)
    names=json.loads((source/"roi_labels.json").read_text())
    manifest=pd.read_csv(manifest_path)
    from importlib.resources import files
    ann=pd.read_csv(files("synergy.resources").joinpath("aal90_annotations.csv"))
    (out/"roi_labels.json").write_text(json.dumps(names))
    if names!=ann.roi_label.tolist():raise ValueError("ROI order mismatch")
    ann["macro_system"]=[_macro_system_from_base_label(x) for x in ann.aal_base_label]
    ann[["roi_label","macro_system"]].to_csv(out/"aal90_macro_system_mapping.csv",index=False)
    subject_rows=[];nodal_rows=[];macro_rows=[];graph_rows=[];mean_matrices={}
    for row in manifest.itertuples(index=False):
        path=source/"subject_matrices"/(row.subject_stub+".npz")
        if not path.exists():raise FileNotFoundError(path)
        with np.load(path,allow_pickle=False) as z:
            sc=z["sc"];fc=z["fc"];mask=z["valid_edge"]
            sc_density=float(np.count_nonzero(sc[U]>0)/len(U[0]))
            for method in ("mmi","ccs"):
                rtr=z[f"{method}_rtr"];sts=z[f"{method}_sts"]
                g=grad(sts,rtr)
                for j,value in enumerate(g):
                    nodal_rows.append(dict(method=method,subject_id=row.subject_id,cohort=row.cohort,
                        stage=row.stage,sedation=row.sedation,roi_index=j+1,roi_label=names[j],
                        include_in_cortical_only=int(ann.include_in_cortical_only.iloc[j]),
                        gradient_value=value))
                for system in SYSTEMS:
                    vals=g[ann.macro_system.to_numpy()==system]
                    macro_rows.append(dict(method=method,subject_id=row.subject_id,cohort=row.cohort,
                        stage=row.stage,sedation=row.sedation,macro_system=system,
                        macro_gradient_mean=float(np.nanmean(vals))))
                subject=dict(method=method,subject_id=row.subject_id,cohort=row.cohort,
                             stage=row.stage,sedation=row.sedation,sc_density=sc_density,
                             n_valid_edges=int(mask[U].sum()))
                for atom,mat in (("rtr",rtr),("sts",sts)):
                    raw=mat[U];good=mask[U]&np.isfinite(raw)
                    positive=int(np.count_nonzero(raw[good]>0))
                    target=int(np.floor(sc_density*len(U[0])))
                    thr=threshold_top_density(mat,sc_density)
                    subject[f"{atom}_positive_edges"]=positive
                    subject[f"{atom}_target_edges"]=target
                    subject[f"{atom}_achieved_edges"]=int(np.count_nonzero((thr[U]>0)&good))
                    subject[f"fc_vs_{atom}_rho"]=rho(fc,mat,mask)
                    subject[f"sc_vs_{atom}_rho"]=rho(sc,mat,mask) # primary signed policy
                    subject[f"sc_vs_{atom}_positive_density_rho"]=rho(sc,thr,mask)
                    subject[f"mean_{atom}"]=float(np.mean(raw[good]))
                    valid_roi=np.isfinite(np.diag(mat))
                    positive_mat=np.clip(mat[np.ix_(valid_roi,valid_roi)],0,None)
                    np.fill_diagonal(positive_mat,0)
                    try: modularity=float(weighted_modularity(positive_mat))
                    except Exception: modularity=np.nan
                    graph_rows.append(dict(method=method,subject_id=row.subject_id,cohort=row.cohort,
                        atom=atom,n_valid_roi=int(valid_roi.sum()),negative_weight_policy="drop negative edges",
                        global_efficiency=float(weighted_global_efficiency(positive_mat)),modularity=modularity))
                    mean_matrices.setdefault((method,row.cohort,atom),[]).append(mat)
                subject_rows.append(subject)
    subject_df=pd.DataFrame(subject_rows)
    nodal_df=pd.DataFrame(nodal_rows)
    macro_df=pd.DataFrame(macro_rows)
    graph_df=pd.DataFrame(graph_rows)
    subject_df.to_csv(out/"subject_atom_similarity.csv",index=False)
    nodal_df.to_csv(out/"subject_nodal_gradients.csv",index=False)
    macro_df.to_csv(out/"subject_macro_gradients.csv",index=False)
    graph_df.to_csv(out/"subject_positive_graph_metrics.csv",index=False)
    # Exact reference figure input: rank of cohort mean matrices, with missing
    # edges omitted from the cohort average rather than treated as zeros.
    mean_rows=[]
    for method in ("mmi","ccs"):
        for cohort in COHORTS:
            with np.errstate(invalid="ignore"):
                sts=np.nanmean(np.array(mean_matrices[(method,cohort,"sts")]),axis=0)
                rtr=np.nanmean(np.array(mean_matrices[(method,cohort,"rtr")]),axis=0)
            g=grad(sts,rtr)
            for j,value in enumerate(g):
                mean_rows.append(dict(method=method,cohort=cohort,roi_index=j+1,roi_label=names[j],gradient_value=value))
    means=pd.DataFrame(mean_rows)
    for method in ("mmi","ccs"):
        folder=out/method/"tables";folder.mkdir(parents=True,exist_ok=True)
        means[means.method==method].to_csv(folder/"cohort_nodal_gradients.csv",index=False)
    # Tests match the visual reference's subject-level unit and FDR families.
    within=[];i=0
    for (method,cohort,system),group in macro_df.groupby(["method","cohort","macro_system"]):
        vals=group.macro_gradient_mean.to_numpy(float)
        p=_perm_test_one_sample(vals,n_perm=10000,seed=14200+i)[1]
        within.append(dict(method=method,cohort=cohort,macro_system=system,
                           mean_gradient=float(np.nanmean(vals)),n_subjects=int(np.isfinite(vals).sum()),perm_p=p))
        i+=1
    within=pd.DataFrame(within)
    within["perm_q_fdr"]=bh(within.perm_p)
    within["significant_fdr"]=within.perm_q_fdr<.05
    within.to_csv(out/"within_system_one_sample_permutation_tests.csv",index=False)
    surface=_surface_gradient_roi_tests(nodal_df,n_perm=10000,seed=15000)
    surface.to_csv(out/"surface_gradient_roi_one_sample_tests.csv",index=False)
    fc_tests=_subject_similarity_tests(subject_df,measure_name="fc_similarity",rtr_col="fc_vs_rtr_rho",
                                       sts_col="fc_vs_sts_rho",n_perm=10000,seed=1200)
    sc_tests=_subject_similarity_tests(subject_df,measure_name="sc_signed_similarity",rtr_col="sc_vs_rtr_rho",
                                       sts_col="sc_vs_sts_rho",n_perm=10000,seed=1300)
    fc_tests.to_csv(out/"fc_similarity_paired_tests.csv",index=False)
    sc_tests.to_csv(out/"sc_signed_similarity_paired_tests.csv",index=False)
    graph_tests=_subject_graph_tests(graph_df,n_perm=10000,seed=1400)
    graph_tests.to_csv(out/"positive_graph_paired_tests.csv",index=False)
    sensitivity=[]
    control=macro_df[macro_df.cohort=="control"]
    for method in ("mmi","ccs"):
        for system in SYSTEMS:
            x=control[(control.method==method)&(control.macro_system==system)].macro_gradient_mean.to_numpy(float)
            for label,coma in (("all_coma",macro_df[macro_df.cohort=="coma"]),
                               ("non_sedated_coma",macro_df[(macro_df.cohort=="coma")&(macro_df.sedation=="non_sedated")]),
                               ("non_sedated_coma_without_roi86_subject",macro_df[(macro_df.cohort=="coma")&(macro_df.sedation=="non_sedated")&(macro_df.subject_id!="coma_acute_non_sedated_idx004")])):
                y=coma[(coma.method==method)&(coma.macro_system==system)].macro_gradient_mean.to_numpy(float)
                diff,p,lo,hi=perm_diff(x,y,seed=3300+len(sensitivity))
                sensitivity.append(dict(method=method,macro_system=system,contrast=label,n_control=len(x),n_coma=len(y),
                                        coma_minus_control=diff,perm_p=p,bootstrap_95_low=lo,bootstrap_95_high=hi))
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity["q_within_contrast"]=sensitivity.groupby("contrast").perm_p.transform(lambda p:bh(p.to_numpy()))
    sensitivity.to_csv(out/"coma_control_sedation_and_roi86_sensitivity.csv",index=False)
    cohort_pairwise_tests(macro_df).to_csv(out/"cohort_macro_pairwise_tests.csv",index=False)
    if not render:
        return out
    fig=out/"figures";fig.mkdir(exist_ok=True)
    render_violin_grid(subject_macro_csv=out/"subject_macro_gradients.csv",
        within_tests_csv=out/"within_system_one_sample_permutation_tests.csv",output_dir=fig/"gradient_stats",
        data_root=out,mapping_csv=out/"aal90_macro_system_mapping.csv")
    render_surface_panel(data_root=out,results_root=out,sig_table=out/"surface_gradient_roi_one_sample_tests.csv",
                         output_dir=fig/"gradient_surface")
    _plot_similarity_paired(subject_df,fc_tests,rtr_col="fc_vs_rtr_rho",sts_col="fc_vs_sts_rho",
        title="FC similarity",ylabel="Spearman rho",out_dir=fig/"fc_similarity",stem="subject_fc_similarity_mmi_vs_ccs_by_cohort_paired_stats")
    _plot_similarity_paired(subject_df,sc_tests,rtr_col="sc_vs_rtr_rho",sts_col="sc_vs_sts_rho",
        title="SC similarity (signed atoms)",ylabel="Spearman rho",out_dir=fig/"sc_similarity",
        stem="subject_sc_signed_similarity_mmi_vs_ccs_by_cohort_paired_stats")
    _plot_graph_paired(graph_df,graph_tests,out_dir=fig/"graph_metrics",stem="subject_positive_graph_metrics_mmi_vs_ccs_paired_stats")
    print(f"Analysis and figures complete: {out}",flush=True)
