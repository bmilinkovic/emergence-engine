"""Publication figure styling for the standalone Synergy package.

The atom colours are semantic: blue is redundancy and red is synergy. Neutral
figure styling is kept separate so that the data colours remain unchanged.
"""

from functools import wraps

import matplotlib as mpl

INK = "#24343A"
MUTED = "#64777B"
PAPER = "#FFFFFF"
ATOM_COLOURS = {"rtr": "#2E5EAA", "sts": "#D43D2A", "balance": "#8A5A44"}

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.titleweight": "semibold",
    "axes.labelsize": 19,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


def styled(function):
    """Apply style to one figure without altering global notebook settings."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with mpl.rc_context(RC):
            return function(*args, **kwargs)
    return wrapped


def finish_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.grid(False)
