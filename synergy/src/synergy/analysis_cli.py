"""Command line entry point for independent DoC analysis and figure rendering."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
from .analysis.doc import run as analyse_doc
from .plotting.pairwise_panels import plot_similarity_paired, plot_graph_paired
from .plotting.gradient_violin import render_violin_grid
from .plotting.gradient_surface import render_surface_panel
from .plotting import atlas_surfaces as atlas
from .plotting.regional_surfaces import plot_doc
from .plotting.export import collect_flat_pdfs


def render(input_dir: Path, output_dir: Path, *, tables_dir: Path | None = None,
           atlas_path: Path, fsaverage_path: Path, mesh_cache: Path) -> None:
    """Render all six figure families from previously computed tables and matrices."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    tables_dir = Path(tables_dir) if tables_dir else input_dir
    fig = output_dir / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    subject = pd.read_csv(tables_dir / "subject_atom_similarity.csv")
    fc = pd.read_csv(tables_dir / "fc_similarity_paired_tests.csv")
    sc = pd.read_csv(tables_dir / "sc_signed_similarity_paired_tests.csv")
    graph = pd.read_csv(tables_dir / "subject_positive_graph_metrics.csv")
    graph_tests = pd.read_csv(tables_dir / "positive_graph_paired_tests.csv")
    plot_similarity_paired(subject, fc, rtr_col="fc_vs_rtr_rho", sts_col="fc_vs_sts_rho",
        title="FC similarity", ylabel="Spearman rho", out_dir=fig / "fc_similarity",
        stem="subject_fc_similarity_mmi_vs_ccs_by_cohort_paired_stats")
    plot_similarity_paired(subject, sc, rtr_col="sc_vs_rtr_rho", sts_col="sc_vs_sts_rho",
        title="SC similarity (signed atoms)", ylabel="Spearman rho", out_dir=fig / "sc_similarity",
        stem="subject_sc_signed_similarity_mmi_vs_ccs_by_cohort_paired_stats")
    plot_graph_paired(graph, graph_tests, out_dir=fig / "graph_metrics",
        stem="subject_positive_graph_metrics_mmi_vs_ccs_paired_stats")
    render_violin_grid(subject_macro_csv=tables_dir / "subject_macro_gradients.csv",
        within_tests_csv=tables_dir / "within_system_one_sample_permutation_tests.csv",
        output_dir=fig / "gradient_stats", data_root=tables_dir,
        mapping_csv=tables_dir / "aal90_macro_system_mapping.csv")
    render_surface_panel(data_root=tables_dir, results_root=tables_dir,
        sig_table=tables_dir / "surface_gradient_roi_one_sample_tests.csv",
        output_dir=fig / "gradient_surface")
    atlas.configure(atlas_path, fsaverage_path, mesh_cache)
    atlas.setup_style()
    meshes = {hemi: atlas.load_surface(hemi) for hemi in ("lh", "rh")}
    plot_doc(input_dir.name, meshes, input_dir, output_dir, mode="all")
    collect_flat_pdfs(fig, output_dir / "all_figures", input_dir.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("render", "analyse", "analyze"))
    p.add_argument("--input", type=Path, required=True,
                   help="Repaired DoC direct/ or deconvolved/ directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, help="Original subject manifest; required for analyse")
    p.add_argument("--atlas", type=Path, default=os.environ.get("PHIID_AAL_ATLAS"))
    p.add_argument("--fsaverage", type=Path, default=os.environ.get("PHIID_FSAVERAGE"))
    p.add_argument("--mesh-cache", type=Path, default=None)
    args = p.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        p.error(f"Input directory does not exist: {source}")
    if args.atlas is None or args.fsaverage is None:
        p.error("Pass --atlas and --fsaverage (or PHIID_AAL_ATLAS and PHIID_FSAVERAGE)")
    if args.command in ("analyse", "analyze"):
        if args.manifest is None:
            p.error("--manifest is required for analyse")
        analyse_doc(source, output, args.manifest, render=False)
    render(source, output, tables_dir=output if args.command in ("analyse", "analyze") else source,
           atlas_path=args.atlas, fsaverage_path=args.fsaverage,
           mesh_cache=args.mesh_cache or output / "surface_cache")


if __name__ == "__main__":
    main()
