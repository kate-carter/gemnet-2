#!/usr/bin/env python3
"""
Figure 2 – Correlation structure landscape across dimensionalities.

Panels:
  A  Eigenvalue spectra (top-50 eigenvalues, one curve per dim, averaged over concepts)
  B  Correlation-value histograms per dim (stacked violins)
  C  Effective rank vs dimensionality per concept
  D  Mean absolute off-diagonal correlation vs dimensionality
  E  Sparsity at each threshold (fraction of zeros) across dims
  F  Heatmap of a representative concept's correlation matrix (256-d)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as sp_stats
from vis_config import (
    apply_style, savefig, CONCEPTS, CONCEPT_COLORS, DIM_COLORS,
    CORR_DIR, MATRICES_DIR, OUTPUT_DIR, DIMS, METHODS,
)

apply_style()

MAX_EIGEN = 50


def _load_corr_upper(concept, dim, corr_dir=None, corr_suffix=None):
    """Load correlation matrix, return (M, upper-triangle values) or None."""
    if corr_dir is None:
        corr_dir = CORR_DIR
    if corr_suffix is None:
        corr_suffix = "spearman"
    path = corr_dir / f"{concept}_{dim}_{corr_suffix}.csv"
    if not path.exists():
        return None
    M = np.loadtxt(path, delimiter=",")
    iu = np.triu_indices(M.shape[0], k=1)
    return M, M[iu]


def _discover_available_dims(corr_dir=None, corr_suffix=None):
    """Return dims that have at least one readable corr matrix."""
    if corr_dir is None:
        corr_dir = CORR_DIR
    if corr_suffix is None:
        corr_suffix = "spearman"
    available = []
    for dim in DIMS:
        for concept in CONCEPTS:
            path = corr_dir / f"{concept}_{dim}_{corr_suffix}.csv"
            if path.exists():
                available.append(dim)
                break
    return available


def _effective_rank(eigenvalues):
    """Shannon-entropy effective rank: exp(H(p)) where p_i = λ_i / Σλ."""
    lam = eigenvalues[eigenvalues > 0]
    p = lam / lam.sum()
    H = -np.sum(p * np.log(p))
    return np.exp(H)


def plot_correlation_landscape(method_name=None, corr_dir=None,
                               matrices_dir=None, corr_suffix=None):
    """Plot the 6-panel correlation landscape for one method.

    Called with no arguments uses the default Spearman paths (backward-compatible).
    """
    if corr_dir is None:
        corr_dir = CORR_DIR
    if matrices_dir is None:
        matrices_dir = MATRICES_DIR
    if corr_suffix is None:
        corr_suffix = "spearman"
    label = method_name if method_name else "Spearman"

    active_dims = _discover_available_dims(corr_dir, corr_suffix)
    if not active_dims:
        print(f"  No {label} correlation matrices found; skipping correlation landscape.")
        return

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"Feature-Correlation Structure — {label}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    ax_eigen    = fig.add_subplot(gs[0, 0])
    ax_hist     = fig.add_subplot(gs[0, 1])
    ax_erank    = fig.add_subplot(gs[0, 2])
    ax_meancorr = fig.add_subplot(gs[1, 0])
    ax_sparsity = fig.add_subplot(gs[1, 1])
    ax_heatmap  = fig.add_subplot(gs[1, 2])

    erank_data      = {d: [] for d in active_dims}
    meancorr_data   = {d: [] for d in active_dims}
    meancorr_concs  = {d: [] for d in active_dims}

    # ── A: Eigenvalue spectra  +  collect data for C & D ────────────────────
    for dim in active_dims:
        all_eigs = []
        color = DIM_COLORS[dim]
        for concept in CONCEPTS:
            result = _load_corr_upper(concept, dim, corr_dir, corr_suffix)
            if result is None:
                continue
            M, upper = result
            eigs = np.linalg.eigvalsh(M)[::-1]
            all_eigs.append(eigs[:MAX_EIGEN])
            erank_data[dim].append(_effective_rank(eigs[eigs > 0]))
            meancorr_data[dim].append(np.mean(np.abs(upper)))
            meancorr_concs[dim].append(concept)

        if all_eigs:
            eig_arr  = np.array(all_eigs)
            mean_eig = eig_arr.mean(axis=0)
            se_eig   = eig_arr.std(axis=0) / np.sqrt(len(all_eigs))
            x = np.arange(1, MAX_EIGEN + 1)
            ax_eigen.plot(x, mean_eig, color=color, linewidth=1.8, label=f"{dim}-d")
            ax_eigen.fill_between(x, mean_eig - se_eig, mean_eig + se_eig,
                                  color=color, alpha=0.15)

    ax_eigen.set_xlabel("Eigenvalue rank")
    ax_eigen.set_ylabel("Eigenvalue")
    ax_eigen.set_title("A. Eigenvalue Spectra (mean ± SE)", fontweight="bold")
    ax_eigen.legend(fontsize=8)
    ax_eigen.set_yscale("log")

    # ── B: Correlation-value distributions ──────────────────────────────────
    violin_data, violin_labels = [], []
    rng = np.random.default_rng(42)
    for dim in active_dims:
        pool = []
        for concept in CONCEPTS:
            result = _load_corr_upper(concept, dim, corr_dir, corr_suffix)
            if result is None:
                continue
            _, upper = result
            sample = rng.choice(upper, size=min(50_000, len(upper)), replace=False)
            pool.append(sample)
        if pool:
            violin_data.append(np.concatenate(pool))
            violin_labels.append(f"{dim}-d")

    if violin_data:
        parts = ax_hist.violinplot(violin_data, showmedians=True, showextrema=False)
        for body, dim in zip(parts["bodies"], active_dims):
            body.set_facecolor(DIM_COLORS[dim])
            body.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        ax_hist.set_xticks(range(1, len(violin_labels) + 1))
        ax_hist.set_xticklabels(violin_labels)
        ks_lines = []
        for i in range(len(violin_data)):
            for j in range(i + 1, len(violin_data)):
                _, p = sp_stats.ks_2samp(violin_data[i], violin_data[j])
                ks_lines.append(f"{violin_labels[i]}–{violin_labels[j]}: p={p:.2e}")
        if ks_lines:
            ax_hist.text(0.02, 0.98, "KS tests:\n" + "\n".join(ks_lines),
                         transform=ax_hist.transAxes, fontsize=6.5,
                         verticalalignment="top", family="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax_hist.set_title("B. Correlation Value Distributions", fontweight="bold")
    ax_hist.set_ylabel("Correlation coefficient")

    # ── C: Effective rank ────────────────────────────────────────────────────
    for dim in active_dims:
        if erank_data[dim]:
            x_jit = dim + np.random.default_rng(7).normal(0, dim * 0.02, len(erank_data[dim]))
            ax_erank.scatter(x_jit, erank_data[dim], color=DIM_COLORS[dim],
                             alpha=0.6, s=30, edgecolors="white", linewidths=0.3)
            ax_erank.scatter([dim], [np.mean(erank_data[dim])], color=DIM_COLORS[dim],
                             s=100, marker="D", edgecolors="black", linewidths=0.8, zorder=5)

    ax_erank.set_xlabel("Embedding Dimensionality")
    ax_erank.set_ylabel("Effective Rank")
    ax_erank.set_xticks(active_dims)

    erank_groups = [erank_data[d] for d in active_dims if erank_data[d]]
    if len(erank_groups) >= 2:
        H, kw_p = sp_stats.kruskal(*erank_groups)
        pw_lines = [f"Kruskal-Wallis: H={H:.1f}, p={kw_p:.2e}"]
        for i in range(len(active_dims)):
            for j in range(i + 1, len(active_dims)):
                if erank_data[active_dims[i]] and erank_data[active_dims[j]]:
                    U, mw_p = sp_stats.mannwhitneyu(
                        erank_data[active_dims[i]], erank_data[active_dims[j]],
                        alternative="two-sided")
                    pw_lines.append(f"{active_dims[i]}–{active_dims[j]}: U={U:.0f}, p={mw_p:.2e}")
        ax_erank.text(0.02, 0.98, "\n".join(pw_lines), transform=ax_erank.transAxes,
                      fontsize=6.5, verticalalignment="top", family="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax_erank.set_title("C. Effective Rank of Correlation Matrix", fontweight="bold")

    # ── D: Mean |correlation| ────────────────────────────────────────────────
    for dim in active_dims:
        if meancorr_data[dim]:
            for i, concept in enumerate(meancorr_concs[dim]):
                ax_meancorr.scatter(dim, meancorr_data[dim][i],
                                    color=CONCEPT_COLORS[concept], alpha=0.6,
                                    s=25, edgecolors="white", linewidths=0.3)
            ax_meancorr.scatter([dim], [np.mean(meancorr_data[dim])],
                                color="black", s=80, marker="D",
                                edgecolors="black", linewidths=0.8, zorder=5)

    ax_meancorr.set_xlabel("Embedding Dimensionality")
    ax_meancorr.set_ylabel("Mean |off-diagonal correlation|")
    ax_meancorr.set_xticks(active_dims)

    mc_groups = [meancorr_data[d] for d in active_dims if meancorr_data[d]]
    if len(mc_groups) >= 2:
        H, kw_p = sp_stats.kruskal(*mc_groups)
        pw_lines = [f"Kruskal-Wallis: H={H:.1f}, p={kw_p:.2e}"]
        for i in range(len(active_dims)):
            for j in range(i + 1, len(active_dims)):
                if meancorr_data[active_dims[i]] and meancorr_data[active_dims[j]]:
                    U, mw_p = sp_stats.mannwhitneyu(
                        meancorr_data[active_dims[i]], meancorr_data[active_dims[j]],
                        alternative="two-sided")
                    pw_lines.append(f"{active_dims[i]}–{active_dims[j]}: U={U:.0f}, p={mw_p:.2e}")
        ax_meancorr.text(0.02, 0.98, "\n".join(pw_lines), transform=ax_meancorr.transAxes,
                         fontsize=6.5, verticalalignment="top", family="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax_meancorr.set_title("D. Mean Absolute Correlation", fontweight="bold")

    # ── E: Sparsity after thresholding ───────────────────────────────────────
    sample_thresholds = [0.05, 0.15, 0.30, 0.50, 0.70]
    for dim in active_dims:
        sparsity = []
        for t in sample_thresholds:
            vals = []
            for concept in CONCEPTS:
                path = matrices_dir / f"{concept}_{dim}_{t:.2f}.csv"
                if not path.exists():
                    path = matrices_dir / f"{concept}_{dim}_{t:.1f}.csv"
                if path.exists():
                    M = np.loadtxt(path, delimiter=",")
                    vals.append(np.sum(M == 0) / M.size)
            sparsity.append(np.mean(vals) if vals else np.nan)
        ax_sparsity.plot(sample_thresholds, sparsity, marker="o",
                         color=DIM_COLORS[dim], linewidth=1.8, label=f"{dim}-d mean")

    ax_sparsity.set_xlabel("Threshold")
    ax_sparsity.set_ylabel("Fraction of zero entries")
    ax_sparsity.set_title("E. Matrix Sparsity vs Threshold", fontweight="bold")
    ax_sparsity.legend(fontsize=8)

    # ── F: Representative heatmap (lowest available dim) ────────────────────
    heatmap_result = heatmap_concept = heatmap_dim = None
    for dim in active_dims:
        for concept in CONCEPTS:
            result = _load_corr_upper(concept, dim, corr_dir, corr_suffix)
            if result is not None:
                heatmap_result, heatmap_concept, heatmap_dim = result, concept, dim
                break
        if heatmap_result is not None:
            break

    if heatmap_result is not None:
        M, _ = heatmap_result
        cmap = LinearSegmentedColormap.from_list("corr", ["#2166ac", "#f7f7f7", "#b2182b"])
        im = ax_heatmap.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax_heatmap.set_title(f"F. {label} Matrix – {heatmap_concept} ({heatmap_dim}-d)",
                             fontweight="bold")
        ax_heatmap.set_xlabel("Feature dimension")
        ax_heatmap.set_ylabel("Feature dimension")
        plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label="Value")

    out_name = (f"correlation_landscape_{method_name}.png"
                if method_name else "correlation_landscape.png")
    savefig(fig, out_name)


def plot_concept_comparison_heatmap(method_name=None, corr_dir=None,
                                    corr_suffix=None):
    """Heatmap: concepts × dims, colored by effective rank (normalized by dim)."""
    if corr_dir is None:
        corr_dir = CORR_DIR
    if corr_suffix is None:
        corr_suffix = "spearman"
    label = method_name if method_name else "Spearman"

    active_dims = _discover_available_dims(corr_dir, corr_suffix)
    if not active_dims:
        print(f"  No {label} corr matrices found; skipping erank heatmap.")
        return

    data = np.full((len(CONCEPTS), len(active_dims)), np.nan)
    for j, dim in enumerate(active_dims):
        for i, concept in enumerate(CONCEPTS):
            result = _load_corr_upper(concept, dim, corr_dir, corr_suffix)
            if result is None:
                continue
            M, _ = result
            eigs  = np.linalg.eigvalsh(M)
            erank = _effective_rank(eigs[eigs > 0])
            data[i, j] = erank / dim

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(active_dims)))
    ax.set_xticklabels([f"{d}-d" for d in active_dims])
    ax.set_yticks(range(len(CONCEPTS)))
    ax.set_yticklabels(CONCEPTS)
    ax.set_title(f"Normalized Effective Rank (erank / dim) — {label}", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()

    out_name = (f"concept_erank_heatmap_{method_name}.png"
                if method_name else "concept_erank_heatmap.png")
    savefig(fig, out_name)


def plot_all_methods_correlation_landscape():
    """Run correlation landscape for every method in METHODS."""
    for method_name, cfg in METHODS.items():
        print(f"  Correlation landscape — {method_name}")
        plot_correlation_landscape(
            method_name=method_name,
            corr_dir=cfg["corr_dir"],
            matrices_dir=cfg["matrices_dir"],
            corr_suffix=cfg["corr_suffix"],
        )


def plot_all_methods_erank_heatmap():
    """Run erank heatmap for every method in METHODS."""
    for method_name, cfg in METHODS.items():
        print(f"  eRank heatmap — {method_name}")
        plot_concept_comparison_heatmap(
            method_name=method_name,
            corr_dir=cfg["corr_dir"],
            corr_suffix=cfg["corr_suffix"],
        )


if __name__ == "__main__":
    plot_all_methods_correlation_landscape()
    plot_all_methods_erank_heatmap()
    print("Done – correlation landscape figures.")
