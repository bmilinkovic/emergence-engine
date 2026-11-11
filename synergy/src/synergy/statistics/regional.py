"""Regional and global tests copied from the original DoC analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .doc import bh,perm_diff
COHORTS=("control","emcs","mcs","uws","coma")
METHODS=("mmi","ccs")

def regional_tests(nodal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for metric in ("rank_gradient", "balance", "sts", "rtr"):
            for cohort in COHORTS[1:]:
                for roi in range(1, 91):
                    group = nodal[(nodal.method == method) & (nodal.metric == metric) &
                                  (nodal.roi_index == roi)]
                    control = group[group.cohort == "control"].value.dropna().to_numpy(float)
                    patient = group[group.cohort == cohort].value.dropna().to_numpy(float)
                    if len(control) >= 2 and len(patient) >= 2:
                        p = float(stats.ttest_ind(patient, control, equal_var=False).pvalue)
                        diff = float(patient.mean() - control.mean())
                    else:
                        p, diff = np.nan, np.nan
                    rows.append(dict(method=method, metric=metric, cohort=cohort,
                                     roi_index=roi, roi_label=group.roi_label.iloc[0],
                                     n_control=len(control), n_patient=len(patient),
                                     patient_minus_control=diff, welch_p=p))
    frame = pd.DataFrame(rows)
    frame["q_fdr_360_per_method_metric"] = frame.groupby(
        ["method", "metric"]).welch_p.transform(lambda p: bh(p.to_numpy(float)))
    return frame


def global_tests(global_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for atom in ("sts", "rtr", "balance"):
            frame = global_df[(global_df.method == method) & (global_df.atom == atom)]
            control = frame[frame.cohort == "control"].mean_edge_value.to_numpy(float)
            for cohort in COHORTS[1:]:
                patient = frame[frame.cohort == cohort].mean_edge_value.to_numpy(float)
                diff, p, lo, hi = perm_diff(control, patient, nperm=10000,
                                            seed=8300+len(rows))
                rows.append(dict(method=method, atom=atom, cohort=cohort,
                                 n_control=len(control), n_patient=len(patient),
                                 patient_minus_control=diff, perm_p=p,
                                 bootstrap_95_low=lo, bootstrap_95_high=hi))
    frame = pd.DataFrame(rows)
    frame["q_fdr_12_per_method"] = frame.groupby("method").perm_p.transform(
        lambda p: bh(p.to_numpy(float)))
    return frame


def coma_global_sensitivity(global_df: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    non_sedated = set(qc[(qc.cohort == "coma") &
                         (qc.sedation == "non_sedated")].subject_id)
    invalid_roi_subjects = set(qc[(qc.cohort == "coma") &
                                  (qc.n_valid_roi < 90)].subject_id)
    rows = []
    for method in METHODS:
        for atom in ("sts", "rtr", "balance"):
            frame = global_df[(global_df.method == method) & (global_df.atom == atom)]
            control = frame[frame.cohort == "control"].mean_edge_value.to_numpy(float)
            coma = frame[frame.cohort == "coma"]
            for label, subjects in (("all_coma", set(coma.subject_id)),
                                    ("non_sedated_coma", non_sedated),
                                    ("non_sedated_without_invalid_roi", non_sedated-invalid_roi_subjects)):
                values = coma[coma.subject_id.isin(subjects)].mean_edge_value.to_numpy(float)
                diff, p, lo, hi = perm_diff(control, values, nperm=10000,
                                            seed=9400+len(rows))
                rows.append(dict(method=method, atom=atom, contrast=label,
                                 n_control=len(control), n_coma=len(values),
                                 coma_minus_control=diff, perm_p=p,
                                 bootstrap_95_low=lo, bootstrap_95_high=hi))
    frame = pd.DataFrame(rows)
    frame["q_fdr_6_per_contrast"] = frame.groupby("contrast").perm_p.transform(
        lambda p: bh(p.to_numpy(float)))
    return frame
