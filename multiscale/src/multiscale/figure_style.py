"""Figure styling shared by the standalone Multiscale plotting functions.

The plotting palette retains the published condition colours. Styling is
scoped to each call so importing this module never changes Matplotlib's global
settings for a user's notebook.
"""

from functools import wraps

import matplotlib as mpl

INK = "#24343A"
MUTED = "#64777B"
PAPER = "#FFFFFF"

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
    """Apply the figure style for one plotting call without leaking rcParams."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with mpl.rc_context(RC):
            return function(*args, **kwargs)
    return wrapped


def finish_axes(ax) -> None:
    """Keep the plotting field clean and leave data marks unobstructed."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.grid(False)
