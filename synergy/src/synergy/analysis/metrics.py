"""Network metrics copied from the original DoC analysis."""
from __future__ import annotations
import numpy as np

def threshold_top_density(
    matrix: np.ndarray,
    density: float,
    *,
    include_diagonal: bool = False,
) -> np.ndarray:
    """Keep the strongest positive edges up to a density of possible edges.

    If fewer positive finite edges exist than the target count, retain all of
    them. The achieved density can therefore be lower than requested.
    """
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {arr.shape}.")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1.")

    n = arr.shape[0]
    tri_k = 0 if include_diagonal else 1
    tri = np.triu_indices(n, k=tri_k)
    values = arr[tri]
    positive = np.where(np.isfinite(values) & (values > 0))[0]
    n_possible = len(values)
    n_keep = min(int(np.floor(density * n_possible)), positive.size)

    out = np.zeros_like(arr)
    if n_keep <= 0:
        return out

    keep_local = positive[np.argsort(values[positive])[-n_keep:]]
    rows = tri[0][keep_local]
    cols = tri[1][keep_local]
    out[rows, cols] = arr[rows, cols]
    out[cols, rows] = arr[rows, cols]
    if include_diagonal:
        np.fill_diagonal(out, np.diag(arr))
    return out


def weighted_global_efficiency(matrix: np.ndarray) -> float:
    """Compute weighted global efficiency using inverse positive weights as distances."""
    weights = np.asarray(matrix, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {weights.shape}.")

    n = weights.shape[0]
    if n < 2:
        return 0.0

    dist = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)
    mask = np.isfinite(weights) & (weights > 0)
    dist[mask] = 1.0 / weights[mask]

    # Floyd-Warshall is appropriate for modest node counts.
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dist
    np.fill_diagonal(inv, 0.0)
    inv[~np.isfinite(inv)] = 0.0
    return float(np.sum(inv) / (n * (n - 1)))


def weighted_modularity(matrix: np.ndarray) -> float:
    """Estimate weighted modularity with a best-effort backend.

    If ``bct`` is available, we use it. Otherwise we fall back to a
    NetworkX-based greedy community estimate. The fallback is useful for planning
    and exploratory work, but it is not numerically identical to the exact
    Brain Connectivity Toolbox path used in the paper.
    """
    weights = np.asarray(matrix, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {weights.shape}.")

    try:
        import bct  # type: ignore

        ci, q = bct.community_louvain(weights)
        _ = ci
        return float(q)
    except Exception:
        pass

    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities, modularity
    except Exception as exc:
        raise RuntimeError(
            "weighted_modularity requires either 'bct' or 'networkx' to be installed."
        ) from exc

    graph = nx.from_numpy_array(np.where(np.isfinite(weights) & (weights > 0), weights, 0.0))
    communities = list(greedy_modularity_communities(graph, weight="weight"))
    if not communities:
        return 0.0
    return float(modularity(graph, communities, weight="weight"))
