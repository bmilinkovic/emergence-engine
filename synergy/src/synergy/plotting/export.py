"""Collect rendered figures under stable publication filenames."""
from __future__ import annotations
from pathlib import Path
from shutil import copy2


def collect_flat_pdfs(figures_dir: Path, output_dir: Path, pipeline: str) -> list[Path]:
    """Copy every DoC figure PDF to an ``all_figures`` collection.

    For example, ``regional_balance/mmi/regional_rank_conditions.pdf`` becomes
    ``DoC__deconvolved__figures__regional_balance__mmi__regional_rank_conditions.pdf``.
    """
    figures_dir = Path(figures_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for source in sorted(figures_dir.rglob("*.pdf")):
        relative = source.relative_to(figures_dir)
        name = "__".join(("DoC", pipeline, "figures", *relative.parts))
        target = destination / name
        copy2(source, target)
        outputs.append(target)
    return outputs
