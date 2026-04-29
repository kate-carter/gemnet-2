"""
vis_method_networks.py — Human vs All Correlation Methods Network Gallery
=========================================================================

Side-by-side network visualizations across all correlation methods.

Layout (n_concepts rows × 6 columns):
  Cols  : Human | Cosine | Spearman | Pearson | Mah. Euclidean | Mah. Gaussian

By default, each LLM network retains the same number of edges as the
corresponding human network for that concept.  Concepts and edge counts
can be overridden via CLI flags.

Usage:
  python vis_method_networks.py                         # defaults
  python vis_method_networks.py --concepts banana car   # custom concepts
  python vis_method_networks.py --edges 500             # fixed edge count
  python vis_method_networks.py --edges 500 200         # per-concept counts

Output: vis_results/method_networks.png
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from vis_config import (
    apply_style, savefig, OUTPUT_DIR,
    CONCEPT_COLORS, METHODS, HUMAN_CSV_DIR,
    load_human_network, BASE,
)

GEMINI2_DIR = BASE / "gemnet-2/gemini-2"

DEFAULT_CONCEPTS = ["grass", "table"]
DIM = 256

CONCEPT_NETWORK_COLORS = {
    "grass": "#3CB371",
    "table": "#2A7AB0",
}

COSINE_MATRICES_DIR = GEMINI2_DIR / "cosine/matrices_cosine256"

COLUMN_ORDER = [
    ("Human",          None),
    ("Cosine",         "Cosine"),
    ("Spearman",       "Spearman"),
    ("Pearson",        "Pearson"),
    ("Mah. Euclidean", "Mah. Euclidean"),
    ("Mah. Gaussian",  "Mah. Gaussian"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_human_edges(concept):
    """Count the number of nonzero edges in the upper triangle of a human network."""
    M, _ = load_human_network(concept)
    np.fill_diagonal(M, 0)
    return int(np.count_nonzero(np.triu(M)))


def _matrix_to_graph(M, top_n):
    """Convert a symmetric matrix to a NetworkX graph, keeping top-N edges."""
    n = M.shape[0]
    np.fill_diagonal(M, 0)
    M = np.triu(M)
    rows, cols = np.nonzero(M)
    weights = M[rows, cols]

    if len(weights) > top_n:
        idx = np.argsort(weights)[-top_n:]
        rows, cols, weights = rows[idx], cols[idx], weights[idx]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for r, c, w in zip(rows, cols, weights):
        if w > 0:
            G.add_edge(r, c, weight=float(w))
    return G


def _draw_network(ax, G, title, color="#4a90d9"):
    """Draw a network on a matplotlib axis."""
    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No edges", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return

    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()

    if H.number_of_nodes() > 500:
        pos = nx.spring_layout(H, k=1.5 / np.sqrt(H.number_of_nodes()),
                               iterations=50, seed=42)
    else:
        try:
            pos = nx.kamada_kawai_layout(H)
        except Exception:
            pos = nx.spring_layout(H, seed=42)

    degrees = dict(H.degree(weight="weight"))
    node_sizes = np.array([degrees.get(n, 0) for n in H.nodes()])
    if node_sizes.max() > 0:
        node_sizes = 5 + 40 * (node_sizes / node_sizes.max())
    else:
        node_sizes = np.full(len(H), 10)

    edge_weights = np.array([d["weight"] for _, _, d in H.edges(data=True)])
    if edge_weights.max() > 0:
        edge_widths = 0.1 + 1.5 * (edge_weights / edge_weights.max())
    else:
        edge_widths = np.full(len(edge_weights), 0.3)

    nx.draw_networkx_edges(H, pos, ax=ax, width=edge_widths,
                           alpha=0.15, edge_color="gray")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=node_sizes,
                           node_color=color, alpha=0.7, edgecolors="white",
                           linewidths=0.3)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def _load_matrix(method_key, concept):
    """Load a 256-d matrix, trying thresholded first then raw correlation."""
    cfg = METHODS[method_key]
    t_str = f"{0.05:.2f}"
    fname = f"{concept}_{DIM}_{t_str}.csv"

    if method_key == "Cosine":
        path = COSINE_MATRICES_DIR / fname
    else:
        path = cfg["matrices_dir"] / fname

    if path.exists():
        return np.loadtxt(path, delimiter=",")

    corr_suffix = cfg.get("corr_suffix", method_key.lower())
    corr_fname = f"{concept}_{DIM}_{corr_suffix}.csv"
    corr_path = cfg["corr_dir"] / corr_fname
    if corr_path.exists():
        return np.loadtxt(corr_path, delimiter=",")

    return None


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_method_networks(concepts=None, edge_counts=None):
    apply_style()

    concepts = concepts or DEFAULT_CONCEPTS

    # Determine per-concept edge counts
    if edge_counts is not None:
        if len(edge_counts) == 1:
            top_n = {c: edge_counts[0] for c in concepts}
        else:
            top_n = {c: n for c, n in zip(concepts, edge_counts)}
    else:
        top_n = {}
        for c in concepts:
            try:
                n = _count_human_edges(c)
                print(f"  {c}: matching human edge count = {n}")
                top_n[c] = n
            except Exception:
                print(f"  {c}: human network not found, defaulting to 200")
                top_n[c] = 200

    n_rows = len(concepts)
    n_cols = len(COLUMN_ORDER)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, concept in enumerate(concepts):
        color = CONCEPT_NETWORK_COLORS.get(concept,
                    CONCEPT_COLORS.get(concept, "#4a90d9"))
        n_edges = top_n[concept]

        for j, (col_label, method_key) in enumerate(COLUMN_ORDER):
            ax = axes[i, j]

            if method_key is None:
                try:
                    M, _ = load_human_network(concept)
                    G = _matrix_to_graph(M, top_n=n_edges)
                    _draw_network(ax, G,
                                  f"n={M.shape[0]}, e={G.number_of_edges()}",
                                  color=color)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="red",
                            wrap=True)
                    ax.axis("off")
            else:
                M = _load_matrix(method_key, concept)
                if M is not None:
                    G = _matrix_to_graph(M, top_n=n_edges)
                    _draw_network(ax, G,
                                  f"{DIM}d, e={G.number_of_edges()}",
                                  color=color)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10, color="gray")
                    ax.axis("off")

    for j, (col_label, _) in enumerate(COLUMN_ORDER):
        axes[0, j].set_title(col_label, fontsize=13, fontweight="bold", pad=12)

    for i, concept in enumerate(concepts):
        axes[i, 0].annotate(
            concept.capitalize(), xy=(-0.2, 0.5), xycoords="axes fraction",
            fontsize=14, fontweight="bold", ha="right", va="center",
            rotation=90,
        )

    fig.suptitle(
        f"Human vs LLM Feature Networks ({DIM}-d)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "method_networks.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method network comparison gallery")
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Concepts to visualize (default: grass table)")
    parser.add_argument("--edges", nargs="+", type=int, default=None,
                        help="Edge count(s): one value for all, or one per concept. "
                             "Default: match human network edge count per concept.")
    args = parser.parse_args()

    plot_method_networks(concepts=args.concepts, edge_counts=args.edges)
    print("Done — method_networks.png")
