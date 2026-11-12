"""Publication figures for DD and structure-fragmentation analyses."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COLORS = ("#2EC4B6", "#5E60CE")
GREEK = {"delta": "δ", "theta": "θ", "alpha": "α",
         "beta": "β", "gamma": "γ"}


def _bands(values):
    observed = list(dict.fromkeys(values))
    return [band for band in GREEK if band in observed] + [
        band for band in observed if band not in GREEK]


def _save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".pdf")


def _stars(q):
    if not np.isfinite(q) or q > .05:
        return ""
    return "***" if q < .001 else "**" if q < .01 else "*"


def _violin(ax, datasets, positions, colors, *, width=.3):
    for values, pos, color in zip(datasets, positions, colors):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        art = ax.violinplot([v], [pos], widths=width, showextrema=False)
        for body in art["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(.42)
        ax.scatter(np.full(len(v), pos), v, s=10, alpha=.32, color=color,
                   edgecolors="none", rasterized=True)
        ax.plot([pos-width*.28, pos+width*.28], [np.median(v)]*2,
                color=color, lw=2)


def plot_dd_violin_by_macro(dd, stats, output, *, conditions=("rest", "dmt"),
                            labels=None, colors=DEFAULT_COLORS):
    """Archived-style condition violins, one panel per macro dimension."""
    bands = _bands(dd.band)
    mvals = sorted(dd.m.unique())
    ncols = 2 if len(mvals) <= 4 else 3
    nrows = int(np.ceil(len(mvals)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows),
                             squeeze=False, sharey=True)
    labels = labels or conditions
    for ax, m in zip(axes.flat, mvals):
        sub = dd[dd.m == m]
        for bi, band in enumerate(bands):
            values = [sub[(sub.band == band) & (sub.condition == cond)].dd_min
                      for cond in conditions]
            positions = [bi-.17, bi+.17]
            _violin(ax, values, positions, colors, width=.3)
            match = stats[(stats.m == m) & (stats.band == band)]
            if len(match):
                star = _stars(float(match.iloc[0].q))
                if star:
                    ax.text(bi, .96, star, transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=16, weight="bold")
        ax.set_title(f"{m}-Macro", fontsize=19)
        ax.set_xticks(range(len(bands)), [GREEK.get(b, b) for b in bands], fontsize=17)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(mvals):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("Minimum dynamical dependence")
    fig.legend([plt.Line2D([], [], marker="o", linestyle="", color=c)
                for c in colors], labels, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, .96, 1))
    return _save(fig, output)


def plot_dd_grand_average(dd, grand_stats, output, *,
                          conditions=("rest", "dmt"), labels=None,
                          colors=DEFAULT_COLORS):
    """Archived pooled-across-macro violin comparison by frequency band."""
    bands = _bands(dd.band)
    fig, ax = plt.subplots(figsize=(15, 6.4))
    for bi, band in enumerate(bands):
        sub = dd[dd.band == band]
        values = [sub[sub.condition == cond].dd_min for cond in conditions]
        _violin(ax, values, [bi-.18, bi+.18], colors, width=.34)
        match = grand_stats[grand_stats.band == band]
        if len(match):
            star = _stars(float(match.iloc[0].q))
            if star:
                ax.text(bi, .97, star, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=22, weight="bold")
    ax.set_xticks(range(len(bands)), [GREEK.get(b, b) for b in bands], fontsize=24)
    ax.set_ylabel("Dynamical dependence", fontsize=22)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    labels = labels or conditions
    ax.legend([plt.Line2D([], [], marker="o", linestyle="", color=c)
               for c in colors], labels, frameon=False, fontsize=13)
    fig.tight_layout()
    return _save(fig, output)


def plot_sf_index(sf, stats, output, *, conditions=("rest", "dmt"),
                  labels=None, colors=DEFAULT_COLORS):
    """SF index by macro dimension and band, with subject trajectories."""
    bands = _bands(sf.band)
    mvals = sorted(sf.m.unique())
    fig, axes = plt.subplots(1, len(bands), figsize=(5.4*len(bands), 6),
                             sharey=True, squeeze=False)
    for ax, band in zip(axes.flat, bands):
        sub = sf[sf.band == band]
        for cond, color in zip(conditions, colors):
            block = sub[sub.condition == cond]
            for _, participant in block.groupby("subject"):
                ax.plot(participant.m, participant.sf_index, color=color,
                        alpha=.13, lw=.7)
            means = block.groupby("m").sf_index.agg(["mean", "std", "count"])
            x = means.index.to_numpy()
            y = means["mean"].to_numpy()
            sem = (means["std"]/np.sqrt(means["count"])).fillna(0).to_numpy()
            ax.plot(x, y, "-o", color=color, lw=2.5)
            ax.fill_between(x, y-1.64*sem, y+1.64*sem,
                            color=color, alpha=.15)
        for m in mvals:
            match = stats[(stats.band == band) & (stats.m == m)]
            if len(match):
                star = _stars(float(match.iloc[0].q))
                if star:
                    ax.text(m, .96, star, transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=15, weight="bold")
        ax.set_title(GREEK.get(band, band), fontsize=25)
        ax.set_xticks(mvals)
        ax.set_ylim(-.03, 1.04)
        ax.grid(alpha=.2)
        ax.set_xlabel("Macro dimension")
    axes[0, 0].set_ylabel("Structure-fragmentation index")
    labels = labels or conditions
    fig.legend([plt.Line2D([], [], color=c, marker="o") for c in colors],
               labels, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, .97, 1))
    return _save(fig, output)


def plot_similarity_matrices(matrices, output, *, conditions=("dmt", "rest"),
                             bands=None, dimensions=None, top_k=40):
    """Archived-style subject-averaged top-40 similarity matrix grid."""
    bands = bands or _bands(k[1] for k in matrices)
    dimensions = dimensions or sorted({k[2] for k in matrices})
    nrows, ncols = len(dimensions), len(bands)*len(conditions)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25*ncols, 2.3*nrows+1),
                             squeeze=False)
    for ri, m in enumerate(dimensions):
        for ci, cond in enumerate(conditions):
            for bi, band in enumerate(bands):
                ax = axes[ri, ci*len(bands)+bi]
                eligible = [g[:top_k, :top_k] for (c, b, mm, _), g in matrices.items()
                            if (c, b, mm) == (cond, band, m) and len(g) >= top_k]
                if eligible:
                    ax.imshow(np.mean(eligible, axis=0), cmap="bone_r",
                              vmin=0, vmax=1, interpolation="nearest")
                    ax.text(.97, .02, f"n={len(eligible)}", ha="right", va="bottom",
                            transform=ax.transAxes, fontsize=8)
                else:
                    ax.set_facecolor("#f3f3f3")
                ax.set_xticks([])
                ax.set_yticks([])
                if ri == 0:
                    ax.set_title(f"{cond}\n{GREEK.get(band, band)}", fontsize=12)
                if ci == 0 and bi == 0:
                    ax.set_ylabel(f"{m}-Macro", fontsize=13)
    fig.suptitle("Mean run similarity distance (top 40 DD solutions)", fontsize=16)
    fig.tight_layout()
    return _save(fig, output)


def plot_timewindow_dd(dd, output, *, rest="rest", rest_label=None, windows=None,
                       colors=DEFAULT_COLORS):
    """Four-window by five-band DD comparison, using paired subject rows."""
    windows = windows or list(dict.fromkeys(dd.window))
    rest_label = rest_label or str(rest).title()
    bands = _bands(dd.band)
    fig, axes = plt.subplots(len(windows), len(bands),
                             figsize=(4.4*len(bands), 3.9*len(windows)),
                             sharey=True, squeeze=False)
    for ri, window in enumerate(windows):
        for ci, band in enumerate(bands):
            ax = axes[ri, ci]
            sub = dd[(dd.window == window) & (dd.band == band)]
            av = sub[sub.condition == rest].groupby("subject").dd_min.mean()
            bv = sub[sub.condition != rest].groupby("subject").dd_min.mean()
            pair = av.to_frame("a").join(bv.to_frame("b"), how="inner")
            _violin(ax, [pair.a, pair.b], [-.2, .2], colors, width=.34)
            ax.set_xticks([-.2, .2], [rest_label, window], rotation=25)
            ax.grid(axis="y", alpha=.2)
            if ri == 0:
                ax.set_title(GREEK.get(band, band), fontsize=18)
            if ci == 0:
                ax.set_ylabel(f"{window}\nDD", fontsize=12)
    fig.tight_layout()
    return _save(fig, output)


def plot_timewindow_sf(sf, output, *, rest="rest", windows=None):
    """Four-window by five-band SF trajectories across macro dimensions."""
    windows = windows or list(dict.fromkeys(sf.window))
    bands = _bands(sf.band)
    fig, axes = plt.subplots(len(windows), len(bands),
                             figsize=(4.4*len(bands), 3.8*len(windows)),
                             sharey=True, squeeze=False)
    for ri, window in enumerate(windows):
        for ci, band in enumerate(bands):
            ax = axes[ri, ci]
            sub = sf[(sf.window == window) & (sf.band == band)]
            for cond, color in zip((rest, next(c for c in sub.condition.unique() if c != rest)),
                                   DEFAULT_COLORS):
                means = sub[sub.condition == cond].groupby("m").sf_index.mean()
                ax.plot(means.index, means.to_numpy(), "-o", color=color, lw=2)
            ax.set_ylim(0, 1)
            ax.set_xticks(sorted(sub.m.unique()))
            ax.grid(alpha=.2)
            if ri == 0:
                ax.set_title(GREEK.get(band, band), fontsize=18)
            if ci == 0:
                ax.set_ylabel(f"{window}\nSF index", fontsize=12)
    fig.tight_layout()
    return _save(fig, output)


def plot_state_trajectory(state_table, value, states, output, *, ylabel=None):
    """Subject paths and group mean ± SEM across ordered states, by band."""
    bands = _bands(state_table.band)
    agg = state_table.groupby(["subject", "state", "band"],
                              as_index=False)[value].mean()
    fig, axes = plt.subplots(1, len(bands), figsize=(5.4*len(bands), 6),
                             sharey=True, squeeze=False)
    for ax, band in zip(axes.flat, bands):
        sub = agg[agg.band == band]
        pivot = sub.pivot(index="subject", columns="state", values=value)
        for _, row in pivot.iterrows():
            y = np.array([row.get(state, np.nan) for state in states])
            ax.plot(range(len(states)), y, color="#888888", alpha=.22, lw=.7)
        summary = sub.groupby("state")[value].agg(["mean", "std", "count"])
        y = np.array([summary.loc[state, "mean"] if state in summary.index
                      else np.nan for state in states])
        sem = np.array([summary.loc[state, "std"] / np.sqrt(summary.loc[state, "count"])
                        if state in summary.index else np.nan for state in states])
        ax.plot(range(len(states)), y, "-o", color="#151515", lw=2.4)
        ax.fill_between(range(len(states)), y-sem, y+sem,
                        color="#151515", alpha=.14)
        ax.set_title(GREEK.get(band, band), fontsize=24)
        ax.set_xticks(range(len(states)), states, rotation=28, ha="right")
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel(ylabel or value)
    fig.tight_layout()
    return _save(fig, output)


def plot_pairwise_heatmap(grand_stats, states, output, *, ylabel=None):
    """Upper-triangle mean state differences with BH-significant stars."""
    bands = _bands(grand_stats.band)
    n = len(states)
    mats = {}
    vmax = 0.0
    for band in bands:
        delta = np.full((n, n), np.nan)
        q = np.full((n, n), np.nan)
        for row in grand_stats[grand_stats.band == band].itertuples():
            i, j = states.index(row.state_a), states.index(row.state_b)
            delta[i, j] = row.delta_mean
            q[i, j] = row.q
            vmax = max(vmax, abs(row.delta_mean))
        mats[band] = delta, q
    fig, axes = plt.subplots(1, len(bands), figsize=(5.4*len(bands), 6),
                             squeeze=False)
    for ax, band in zip(axes.flat, bands):
        delta, q = mats[band]
        im = ax.imshow(np.ma.masked_invalid(delta), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        for i in range(n):
            for j in range(i+1, n):
                if np.isfinite(q[i, j]) and q[i, j] <= .05:
                    ax.text(j, i, "★", ha="center", va="center",
                            color="#F4C542", fontsize=22)
        ax.set_title(GREEK.get(band, band), fontsize=24)
        ax.set_xticks(range(n), states, rotation=45, ha="right")
        ax.set_yticks(range(n), states)
    if ylabel:
        axes[0, 0].set_ylabel(ylabel)
    fig.colorbar(im, ax=axes, shrink=.55, label="Mean difference (column − row)")
    fig.subplots_adjust(left=.06, right=.92, bottom=.20, top=.87, wspace=.14)
    return _save(fig, output)
