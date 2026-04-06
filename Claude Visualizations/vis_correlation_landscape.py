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
    CORR_DIR, MATRICES_DIR, OUTPUT_DIR, DIMS,
)

apply_style()

MAX_EIGEN = 50
SAMPLE_DIMS = [256, 512, 768]  # dims small enough to load fully on a laptop


def _load_corr_upper(concept, dim):
    """Load correlation matrix, return upper-triangle values (no diagonal)."""
    path = CORR_DIR / f"{concept}_{dim}_mahalanobis.csv"
    if not path.exists():
        return None
    M = np.loadtxt(path, delimiter=",")
    iu = np.triu_indices(M.shape[0], k=1)
    return M, M[iu]


def _effective_rank(eigenvalues):
    """Shannon-entropy effective rank: exp(H(p)) where p_i = λ_i / Σλ."""
    lam = eigenvalues[eigenvalues > 0]
    p = lam / lam.sum()
    H = -np.sum(p * np.log(p))
    return np.exp(H)


def plot_correlation_landscape():
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        "Feature-Correlation Structure Across Embedding Dimensionalities",
        fontsize=14, fontweight="bold", y=0.99,
    )

    ax_eigen = fig.add_subplot(gs[0, 0])
    ax_hist  = fig.add_subplot(gs[0, 1])
    ax_erank = fig.add_subplot(gs[0, 2])
    ax_meancorr = fig.add_subplot(gs[1, 0])
    ax_sparsity = fig.add_subplot(gs[1, 1])
    ax_heatmap  = fig.add_subplot(gs[1, 2])

    erank_data = {d: [] for d in SAMPLE_DIMS}
    meancorr_data = {d: [] for d in SAMPLE_DIMS}

    # ── A: Eigenvalue spectra  &  C: Effective rank  &  D: Mean |corr| ─────
    for dim in SAMPLE_DIMS:
        all_eigs = []
        color = DIM_COLORS[dim]

        for concept in CONCEPTS:
            result = _load_corr_upper(concept, dim)
            if result is None:
                continue
            M, upper = result

            eigs = np.linalg.eigvalsh(M)[::-1]
            all_eigs.append(eigs[:MAX_EIGEN])

            erank_data[dim].append(_effective_rank(eigs[eigs > 0]))
            meancorr_data[dim].append(np.mean(np.abs(upper)))

        if all_eigs:
            eig_arr = np.array(all_eigs)
            mean_eig = eig_arr.mean(axis=0)
            se_eig = eig_arr.std(axis=0) / np.sqrt(len(all_eigs))
            x = np.arange(1, MAX_EIGEN + 1)
            ax_eigen.plot(x, mean_eig, color=color, linewidth=1.8, label=f"{dim}-d")
            ax_eigen.fill_between(x, mean_eig - se_eig, mean_eig + se_eig,
                                  color=color, alpha=0.15)

    ax_eigen.set_xlabel("Eigenvalue rank")
    ax_eigen.set_ylabel("Eigenvalue")
    ax_eigen.set_title("A. Eigenvalue Spectra (mean ± SE)", fontweight="bold")
    ax_eigen.legend(fontsize=8)
    ax_eigen.set_yscale("log")

    # ── B: Correlation-value distributions ─────────────────────────────────
    violin_data, violin_labels = [], []
    for dim in SAMPLE_DIMS:
        pool = []
        for concept in CONCEPTS:
            result = _load_corr_upper(concept, dim)
            if result is None:
                continue
            _, upper = result
            rng = np.random.default_rng(42)
            sample = rng.choice(upper, size=min(50_000, len(upper)), replace=False)
            pool.append(sample)
        if pool:
            violin_data.append(np.concatenate(pool))
            violin_labels.append(f"{dim}-d")

    if violin_data:
        parts = ax_hist.violinplot(violin_data, showmedians=True, showextrema=False)
        for i, (body, dim) in enumerate(zip(parts["bodies"], SAMPLE_DIMS)):
            body.set_facecolor(DIM_COLORS[dim])
            body.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        ax_hist.set_xticks(range(1, len(violin_labels) + 1))
        ax_hist.set_xticklabels(violin_labels)

        # Pairwise KS-tests between dims
        ks_lines = []
        for i in range(len(violin_data)):
            for j in range(i + 1, len(violin_data)):
                stat, p = sp_stats.ks_2samp(violin_data[i], violin_data[j])
                ks_lines.append(f"{violin_labels[i]}–{violin_labels[j]}: p={p:.2e}")
        if ks_lines:
            ax_hist.text(0.02, 0.98, "KS tests:\n" + "\n".join(ks_lines),
                         transform=ax_hist.transAxes, fontsize=6.5,
                         verticalalignment="top", family="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax_hist.set_title("B. Correlation Value Distributions", fontweight="bold")
    ax_hist.set_ylabel("Correlation coefficient")

    # ── C: Effective rank ──────────────────────────────────────────────────
    for dim in SAMPLE_DIMS:
        if erank_data[dim]:
            x_jitter = dim + np.random.default_rng(7).normal(0, dim * 0.02, len(erank_data[dim]))
            ax_erank.scatter(
                x_jitter, erank_data[dim],
                color=DIM_COLORS[dim], alpha=0.6, s=30, edgecolors="white", linewidths=0.3,
            )
            ax_erank.scatter(
                [dim], [np.mean(erank_data[dim])],
                color=DIM_COLORS[dim], s=100, marker="D", edgecolors="black",
                linewidths=0.8, zorder=5,
            )
    ax_erank.set_xlabel("Embedding Dimensionality")
    ax_erank.set_ylabel("Effective Rank")
    ax_erank.set_xticks(SAMPLE_DIMS)

    # Kruskal-Wallis across dims + pairwise Mann-Whitney
    erank_groups = [erank_data[d] for d in SAMPLE_DIMS if erank_data[d]]
    if len(erank_groups) >= 2:
        H, kw_p = sp_stats.kruskal(*erank_groups)
        pw_lines = [f"Kruskal-Wallis: H={H:.1f}, p={kw_p:.2e}"]
        for i in range(len(SAMPLE_DIMS)):
            for j in range(i + 1, len(SAMPLE_DIMS)):
                if erank_data[SAMPLE_DIMS[i]] and erank_data[SAMPLE_DIMS[j]]:
                    U, mw_p = sp_stats.mannwhitneyu(
                        erank_data[SAMPLE_DIMS[i]], erank_data[SAMPLE_DIMS[j]],
                        alternative="two-sided")
                    pw_lines.append(f"{SAMPLE_DIMS[i]}–{SAMPLE_DIMS[j]}: U={U:.0f}, p={mw_p:.2e}")
        ax_erank.text(0.02, 0.98, "\n".join(pw_lines),
                      transform=ax_erank.transAxes, fontsize=6.5,
                      verticalalignment="top", family="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax_erank.set_title("C. Effective Rank of Correlation Matrix", fontweight="bold")

    # ── D: Mean |correlation| ──────────────────────────────────────────────
    for dim in SAMPLE_DIMS:
        if meancorr_data[dim]:
            for i, concept in enumerate(CONCEPTS[:len(meancorr_data[dim])]):
                ax_meancorr.scatter(
                    dim, meancorr_data[dim][i],
                    color=CONCEPT_COLORS[concept], alpha=0.6, s=25,
                    edgecolors="white", linewidths=0.3,
                )
            ax_meancorr.scatter(
                [dim], [np.mean(meancorr_data[dim])],
                color="black", s=80, marker="D", edgecolors="black",
                linewidths=0.8, zorder=5,
            )
    ax_meancorr.set_xlabel("Embedding Dimensionality")
    ax_meancorr.set_ylabel("Mean |off-diagonal correlation|")
    ax_meancorr.set_xticks(SAMPLE_DIMS)

    mc_groups = [meancorr_data[d] for d in SAMPLE_DIMS if meancorr_data[d]]
    if len(mc_groups) >= 2:
        H, kw_p = sp_stats.kruskal(*mc_groups)
        pw_lines = [f"Kruskal-Wallis: H={H:.1f}, p={kw_p:.2e}"]
        for i in range(len(SAMPLE_DIMS)):
            for j in range(i + 1, len(SAMPLE_DIMS)):
                if meancorr_data[SAMPLE_DIMS[i]] and meancorr_data[SAMPLE_DIMS[j]]:
                    U, mw_p = sp_stats.mannwhitneyu(
                        meancorr_data[SAMPLE_DIMS[i]], meancorr_data[SAMPLE_DIMS[j]],
                        alternative="two-sided")
                    pw_lines.append(f"{SAMPLE_DIMS[i]}–{SAMPLE_DIMS[j]}: U={U:.0f}, p={mw_p:.2e}")
        ax_meancorr.text(0.02, 0.98, "\n".join(pw_lines),
                         transform=ax_meancorr.transAxes, fontsize=6.5,
                         verticalalignment="top", family="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax_meancorr.set_title("D. Mean Absolute Correlation", fontweight="bold")

    # ── E: Sparsity after thresholding ─────────────────────────────────────
    sample_thresholds = [0.05, 0.15, 0.30, 0.50, 0.70]
    for dim in [256]:  # keep lightweight
        for concept in ["banana"]:
            sparsity = []
            for t in sample_thresholds:
                path = MATRICES_DIR / f"{concept}_{dim}_{t:.2f}.csv"
                if not path.exists():
                    path = MATRICES_DIR / f"{concept}_{dim}_{t:.1f}.csv"
                if path.exists():
                    M = np.loadtxt(path, delimiter=",")
                    frac_zero = np.sum(M == 0) / M.size
                    sparsity.append(frac_zero)
                else:
                    sparsity.append(np.nan)
            ax_sparsity.plot(
                sample_thresholds, sparsity,
                marker="o", color=DIM_COLORS[dim], linewidth=1.8,
                label=f"{concept} ({dim}-d)",
            )
    ax_sparsity.set_xlabel("Threshold")
    ax_sparsity.set_ylabel("Fraction of zero entries")
    ax_sparsity.set_title("E. Matrix Sparsity vs Threshold", fontweight="bold")
    ax_sparsity.legend(fontsize=8)

    # ── F: Heatmap of representative correlation matrix ────────────────────
    result = _load_corr_upper("banana", 256)
    if result is not None:
        M, _ = result
        cmap = LinearSegmentedColormap.from_list(
            "corr", ["#2166ac", "#f7f7f7", "#b2182b"]
        )
        im = ax_heatmap.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax_heatmap.set_title('F. Correlation Matrix – banana (256-d)', fontweight="bold")
        ax_heatmap.set_xlabel("Feature dimension")
        ax_heatmap.set_ylabel("Feature dimension")
        plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label="Correlation")

    savefig(fig, "correlation_landscape.png")


def plot_concept_comparison_heatmap():
    """Heatmap: concepts × dims, colored by effective rank (normalized by dim)."""
    data = np.full((len(CONCEPTS), len(SAMPLE_DIMS)), np.nan)

    for j, dim in enumerate(SAMPLE_DIMS):
        for i, concept in enumerate(CONCEPTS):
            result = _load_corr_upper(concept, dim)
            if result is None:
                continue
            M, _ = result
            eigs = np.linalg.eigvalsh(M)
            erank = _effective_rank(eigs[eigs > 0])
            data[i, j] = erank / dim  # normalized

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(SAMPLE_DIMS)))
    ax.set_xticklabels([f"{d}-d" for d in SAMPLE_DIMS])
    ax.set_yticks(range(len(CONCEPTS)))
    ax.set_yticklabels(CONCEPTS)
    ax.set_title("Normalized Effective Rank (erank / dim)", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    savefig(fig, "concept_erank_heatmap.png")


if __name__ == "__main__":
    print("Computing correlation landscape (loading 256/512/768-d matrices)...")
    plot_correlation_landscape()
    print("Computing concept × dim effective-rank heatmap...")
    plot_concept_comparison_heatmap()
    print("Done – correlation landscape figures.")
