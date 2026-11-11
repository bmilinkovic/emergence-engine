"""Macro-system helpers copied from the original DoC analysis."""
from __future__ import annotations
import numpy as np

def _perm_test_one_sample(x: np.ndarray, *, n_perm: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, arr.size), replace=True)
    perm = (signs * arr[None, :]).mean(axis=1)
    p = (1.0 + np.sum(np.abs(perm) >= abs(obs))) / (n_perm + 1.0)
    return obs, float(p)


def _macro_system_from_base_label(base: str) -> str:
    if base in {"Precentral", "Rolandic_Oper", "Supp_Motor_Area", "Postcentral", "Paracentral_Lobule"}:
        return "Sensorimotor"
    if base in {"Frontal_Sup", "Frontal_Mid", "Frontal_Inf_Oper", "Frontal_Inf_Tri", "Frontal_Sup_Medial"}:
        return "Frontal association"
    if base in {"Frontal_Sup_Orb", "Frontal_Mid_Orb", "Frontal_Inf_Orb", "Frontal_Med_Orb", "Rectus", "Olfactory"}:
        return "Orbitofrontal"
    if base == "Insula":
        return "Insula"
    if base in {"Cingulum_Ant", "Cingulum_Mid", "Cingulum_Post", "Hippocampus", "ParaHippocampal", "Amygdala"}:
        return "Limbic"
    if base in {"Calcarine", "Cuneus", "Lingual", "Occipital_Sup", "Occipital_Mid", "Occipital_Inf", "Fusiform"}:
        return "Occipital-ventral"
    if base in {"Parietal_Sup", "Parietal_Inf", "SupraMarginal", "Angular", "Precuneus"}:
        return "Parietal"
    if base in {"Heschl", "Temporal_Sup", "Temporal_Pole_Sup", "Temporal_Mid", "Temporal_Pole_Mid", "Temporal_Inf"}:
        return "Temporal"
    if base in {"Caudate", "Putamen", "Pallidum", "Thalamus"}:
        return "Subcortical"
    raise KeyError(f"Unhandled AAL base label: {base}")
