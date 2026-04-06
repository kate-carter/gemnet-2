#!/usr/bin/env python3
"""
Figure 5 – Network visualization gallery.

Small-multiple grid showing force-directed layouts of thresholded correlation
networks at representative thresholds for select concepts (256-d).
Also shows human networks side by side for direct visual comparison.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from vis_config import (
    apply_style, savefig, CONCEPTS, CONCEPT_COLORS,
    MATRICES_DIR, HUMAN_CSV_DIR, OUTPUT_DIR,
    load_human_network,
)

apply_style()

GALLERY_CONCEPTS = ["banana", "car", "key", "phone", "wood"]
GALLERY_THRESHOLDS = [0.10, 0.20, 0.35, 0.50]
DIM = 256
TOP_EDGES = 800  # cap edges for clarity


def _matrix_to_graph(M, top_n=TOP_EDGES):
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
        ax.set_title(title, fontsize=9)
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
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")


def plot_llm_network_gallery():
    """Grid: concepts (rows) × thresholds (columns)."""
    n_rows = len(GALLERY_CONCEPTS)
    n_cols = len(GALLERY_THRESHOLDS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(
        f"LLM Feature Networks ({DIM}-d) Across Thresholds",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for i, concept in enumerate(GALLERY_CONCEPTS):
        for j, thresh in enumerate(GALLERY_THRESHOLDS):
            ax = axes[i, j]
            fname = f"{concept}_{DIM}_{thresh:.2f}.csv"
            path = MATRICES_DIR / fname
            if not path.exists():
                ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.axis("off")
                continue
            M = np.loadtxt(path, delimiter=",")
            G = _matrix_to_graph(M)
            title = f"{concept} (t={thresh:.2f})"
            if j == 0:
                title = f"{concept}\nt={thresh:.2f}"
            else:
                title = f"t={thresh:.2f}"
            _draw_network(ax, G, title, color=CONCEPT_COLORS[concept])

    for j, thresh in enumerate(GALLERY_THRESHOLDS):
        axes[0, j].set_title(f"Threshold = {thresh:.2f}", fontsize=11, fontweight="bold")
    for i, concept in enumerate(GALLERY_CONCEPTS):
        axes[i, 0].annotate(
            concept, xy=(-0.15, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", ha="right", va="center",
            rotation=90,
        )

    fig.tight_layout()
    savefig(fig, "network_gallery_llm.png")


def plot_human_network_gallery():
    """Row of human networks for the gallery concepts."""
    n = len(GALLERY_CONCEPTS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    fig.suptitle(
        "Human Feature Networks",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for i, concept in enumerate(GALLERY_CONCEPTS):
        ax = axes[i]
        try:
            M, labels = load_human_network(concept)
            G = _matrix_to_graph(M, top_n=2000)
            _draw_network(ax, G, f"{concept} (n={M.shape[0]})",
                          color=CONCEPT_COLORS[concept])
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
            ax.axis("off")

    fig.tight_layout()
    savefig(fig, "network_gallery_human.png")


def plot_side_by_side():
    """Direct side-by-side: human vs LLM at best-match threshold for each concept."""
    n = len(GALLERY_CONCEPTS)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(
        "Human (top) vs LLM 256-d (bottom) Feature Networks",
        fontsize=14, fontweight="bold", y=1.02,
    )

    best_thresh = 0.15  # representative moderate threshold

    for i, concept in enumerate(GALLERY_CONCEPTS):
        # Human
        ax_h = axes[0, i]
        try:
            M_h, _ = load_human_network(concept)
            G_h = _matrix_to_graph(M_h, top_n=2000)
            _draw_network(ax_h, G_h, f"{concept} (human, n={M_h.shape[0]})",
                          color=CONCEPT_COLORS[concept])
        except Exception:
            ax_h.text(0.5, 0.5, "N/A", ha="center", va="center",
                      transform=ax_h.transAxes)
            ax_h.axis("off")

        # LLM
        ax_l = axes[1, i]
        fname = f"{concept}_{DIM}_{best_thresh:.2f}.csv"
        path = MATRICES_DIR / fname
        if path.exists():
            M_l = np.loadtxt(path, delimiter=",")
            G_l = _matrix_to_graph(M_l)
            _draw_network(ax_l, G_l,
                          f"{concept} (LLM {DIM}-d, t={best_thresh})",
                          color=CONCEPT_COLORS[concept])
        else:
            ax_l.text(0.5, 0.5, "Not found", ha="center", va="center",
                      transform=ax_l.transAxes)
            ax_l.axis("off")

    fig.tight_layout()
    savefig(fig, "network_gallery_sidebyside.png")


if __name__ == "__main__":
    print("Drawing LLM network gallery...")
    plot_llm_network_gallery()
    print("Drawing human network gallery...")
    plot_human_network_gallery()
    print("Drawing side-by-side comparison...")
    plot_side_by_side()
    print("Done – network gallery figures.")
