#!/usr/bin/env python3
"""Publication-style cortical surfaces and companion PhiID plots for Maas within cohorts."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "synergy_mpl"))

import fast_simplification
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.text import Text
from fontTools.ttLib import TTCollection
import nibabel as nib
from nibabel.freesurfer.io import read_geometry, read_morph_data
from nilearn import plotting
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ATLAS: Path | None = None
FSAVG: Path | None = None
SURFACE_CACHE: Path | None = None

def configure(atlas: Path, fsaverage: Path, cache: Path) -> None:
    """Set local AAL atlas, FreeSurfer fsaverage and writable mesh cache."""
    global ATLAS, FSAVG, SURFACE_CACHE
    ATLAS, FSAVG, SURFACE_CACHE = Path(atlas), Path(fsaverage), Path(cache)
    if not ATLAS.exists() or not FSAVG.exists():
        raise FileNotFoundError("AAL atlas and fsaverage surface are required")

INK="#27332f"
GREEN="#174f46"
ORANGE="#e9780d"
MIST="#f8f7f3"
RANK_CMAP=LinearSegmentedColormap.from_list(
    "bone_green_to_burnt_orange",
    ["#123f39","#579380","#c9e6d9","#f8f7f3","#fce5ad","#f4ad35","#e9780d"],
    N=256,
)
DRUG_CHANGE_CMAP=LinearSegmentedColormap.from_list(
    "zissou_signed_change",
    ["#3b9ab2","#78b7c5","#f8f7f3","#f3a18c","#f21a00"],
    N=256,
)
STS_CMAP=LinearSegmentedColormap.from_list(
    "sts_change",["#123f39","#579380","#c9e6d9","#f8f7f3",
                  "#fce5ad","#f4ad35","#e9780d"],N=256)
RTR_CMAP=LinearSegmentedColormap.from_list(
    "rtr_change",["#e9780d","#f4ad35","#fce5ad","#f8f7f3",
                  "#c9e6d9","#579380","#123f39"],N=256)

DISPLAY_NAME={"psil_within":"Psilocybin","thc_within":"THC"}


def setup_style() -> None:
    bold_path=Path(tempfile.gettempdir()) / "synergy_mpl" / "Helvetica-Bold.ttf"
    font_family="Helvetica"
    if not bold_path.exists():
        bold_path.parent.mkdir(parents=True,exist_ok=True)
        mac_font=Path("/System/Library/Fonts/Helvetica.ttc")
        if mac_font.exists():
            TTCollection(str(mac_font)).fonts[1].save(bold_path)
        else:
            font_family="DejaVu Sans"
    if bold_path.exists():
        font_manager.fontManager.addfont(str(bold_path))
    plt.rcParams.update({
        "figure.facecolor":"none","axes.facecolor":"none","savefig.facecolor":"none",
        "font.family":font_family,"font.weight":"bold","font.size":11,
        "axes.titleweight":"bold","axes.labelweight":"bold","text.color":INK,
        "axes.labelcolor":INK,"axes.edgecolor":"#8b958e","xtick.color":INK,
        "ytick.color":INK,"pdf.fonttype":42,
    })


def save(fig: plt.Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf"):
        fig.savefig(folder/f"{name}.{ext}",dpi=260,bbox_inches="tight",
                    transparent=True)
    plt.close(fig)


def sample_cortical_codes(pial: np.ndarray, white: np.ndarray,
                          world_to_atlas: np.ndarray, atlas_data: np.ndarray) -> np.ndarray:
    codes=np.zeros(len(pial),dtype=np.int32)
    for depth in (0.0,.25):
        points=(1-depth)*pial+depth*white
        vox=np.rint(nib.affines.apply_affine(world_to_atlas,points)).astype(int)
        inside=np.all((vox>=0)&(vox<np.asarray(atlas_data.shape)),axis=1)
        sampled=np.zeros(len(points),dtype=np.int32)
        sampled[inside]=atlas_data[tuple(vox[inside].T)]
        cortical=(sampled>0)&((sampled<7000)|((sampled>=8000)&(sampled<9000)))
        use=(codes==0)&cortical
        codes[use]=sampled[use]
    return codes


def load_surface(hemi: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Decimate fsaverage inflated mesh and map its pial vertices to AAL116."""
    if SURFACE_CACHE is None or FSAVG is None or ATLAS is None:
        raise RuntimeError("Call configure() before loading surfaces")
    SURFACE_CACHE.mkdir(parents=True,exist_ok=True)
    cache=SURFACE_CACHE/f"fsaverage_{hemi}_aal116_30000faces.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["vertices"],z["faces"],z["codes"],z["sulc"]
    surf=FSAVG/"surf"
    inflated,faces=read_geometry(str(surf/f"{hemi}.inflated"))
    pial,_=read_geometry(str(surf/f"{hemi}.pial"))
    white,_=read_geometry(str(surf/f"{hemi}.white"))
    sulc=read_morph_data(str(surf/f"{hemi}.sulc"))
    vertices,small_faces=fast_simplification.simplify(
        inflated,faces,target_count=30000)
    original=cKDTree(inflated).query(vertices)[1]
    fs_t1=nib.load(str(FSAVG/"mri/T1.mgz"))
    atlas=nib.load(str(ATLAS))
    world_to_atlas=(np.linalg.inv(atlas.affine)@fs_t1.affine@
                    np.linalg.inv(fs_t1.header.get_vox2ras_tkr()))
    codes=sample_cortical_codes(pial,white,world_to_atlas,
                                np.asarray(atlas.dataobj,dtype=np.int32))[original]
    small_sulc=np.asarray(sulc[original],dtype=np.float32)
    np.savez_compressed(cache,vertices=vertices,faces=small_faces,
                        codes=codes,sulc=small_sulc)
    return vertices,small_faces,codes,small_sulc


def values_on_mesh(values: np.ndarray, roi_names: np.ndarray,
                   codes: np.ndarray) -> np.ndarray:
    code_map={int(name.rsplit("cluster",1)[1]):float(value)
              for name,value in zip(roi_names,values)}
    mapped=np.zeros(len(codes),dtype=float)
    for code,value in code_map.items():
        if np.isfinite(value):mapped[codes==code]=value
    return mapped


def surf_panel(ax: plt.Axes, mesh: tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray],
               values: np.ndarray, roi_names: np.ndarray, *, hemi: str,
               view: str, cmap: LinearSegmentedColormap, bound: float) -> None:
    vertices,faces,codes,sulc=mesh
    mapped=values_on_mesh(values,roi_names,codes)
    plotting.plot_surf_stat_map(
        (vertices,faces),mapped,bg_map=sulc,hemi="left" if hemi=="lh" else "right",
        view=view,axes=ax,figure=ax.figure,cmap=cmap,colorbar=False,
        vmin=-bound,vmax=bound,symmetric_cbar=True,threshold=1e-12,
        bg_on_data=False,darkness=.35,
    )
    ax.patch.set_alpha(0)


def render_surface_rows(dataset: str, roi_names: np.ndarray,
                        rows: list[tuple[str,np.ndarray]], bound: float,
                        filename: str, folder: Path,
                        meshes: dict[str,tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]],
                        *, title: str, unit: str, bar_ends: tuple[str,str],
                        cmap: LinearSegmentedColormap = RANK_CMAP,
                        significant: np.ndarray | None = None) -> None:
    fig=plt.figure(figsize=(12.8,2.8*len(rows)+.75),facecolor="none")
    grid=fig.add_gridspec(len(rows),4,left=.08,right=.92,top=.84,bottom=.12,
                          wspace=-.14,hspace=-.20)
    views=[("lh","lateral"),("lh","medial"),("rh","medial"),("rh","lateral")]
    for row,(label,values) in enumerate(rows):
        for col,(hemi,view) in enumerate(views):
            ax=fig.add_subplot(grid[row,col],projection="3d")
            surf_panel(ax,meshes[hemi],values,roi_names,hemi=hemi,view=view,
                       cmap=cmap,bound=bound)
            row_significant=(significant[row] if significant is not None and
                             np.ndim(significant)==2 else significant)
            if row_significant is not None and np.any(row_significant):
                mapped=values_on_mesh(row_significant.astype(float),roi_names,meshes[hemi][2])
                if np.any(mapped):
                    plotting.plot_surf_contours(
                        (meshes[hemi][0],meshes[hemi][1]),mapped,
                        hemi="left" if hemi=="lh" else "right",levels=[1],
                        colors=["#1b211e"],axes=ax,figure=fig)
        if len(rows)>1:
            display_label="Control" if label=="CONTROL" else label
            fig.text(.018,.84-(row+.5)*(.72/len(rows)),display_label,
                     ha="left",va="center",fontsize=12,weight="bold",color=INK)
    bar_x,bar_y,bar_width,bar_height=.34,.065,.32,.045
    cax=fig.add_axes((bar_x,bar_y,bar_width,bar_height))
    colorbar=fig.colorbar(ScalarMappable(norm=Normalize(-bound,bound),
                                        cmap=cmap),cax=cax,
                          orientation="horizontal")
    colorbar.set_ticks([-bound,0,bound])
    colorbar.set_ticklabels([
        f"{x:.0f}" if bound>=1 else f"{x:.2f}" if bound>=.01 else
        "0" if x==0 else f"{x:.2g}" for x in (-bound,0,bound)
    ])
    colorbar.ax.tick_params(labelsize=13,length=0,pad=4)
    colorbar.outline.set_visible(False)
    fig.text(bar_x+bar_width,bar_y+bar_height+.025,bar_ends[1],
             ha="right",va="bottom",fontsize=13,color=INK)
    if bar_ends[0]:
        fig.text(bar_x,bar_y+bar_height+.025,bar_ends[0],
                 ha="left",va="bottom",fontsize=13,color=INK)
    fig.text(.5,.97,title,fontsize=20,weight="bold",ha="center",va="top",color=INK)
    for label in fig.findobj(match=Text):
        label.set_fontfamily(plt.rcParams["font.family"][0])
        label.set_fontweight("bold")
    save(fig,folder,filename)

