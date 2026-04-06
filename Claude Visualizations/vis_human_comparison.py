#!/usr/bin/env python3
"""
Figure 3 – Human vs LLM (Mahalanobis) network comparison.

Panels:
  A  Radar chart: per-concept metric profile (human vs best-matching LLM threshold)
  B  Weight distribution comparison (human vs LLM at matched threshold)
  C  Degree-distribution CDFs
  D  Metric distance heatmap (concept × threshold)
  E  Bar chart: which threshold best matches human for each metric
  F  Network size comparison (human node count vs LLM dimensionality)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
from vis_config import (
    apply_style, savefig, CONCEPTS, CONCEPT_COLORS,
    RESULTS_DIR, HUMAN_CSV_DIR, OUTPUT_DIR, load_human_network,
)

apply_style()


def _compute_human_stats(concept):
    """Compute basic graph statistics on a human network for comparison."""
    M, labels = load_human_network(concept)
    n = M.shape[0]

    np.fill_diagonal(M, 0)
    strength = M.sum(axis=1)
    degree = (M > 0).sum(axis=1).astype(float)

    # Weighted clustering: for each node, ratio of weighted triangles to possible
    A_bin = (M > 0).astype(float)
    W_norm = M / M.max() if M.max() > 0 else M
    W_third = np.cbrt(W_norm)
    numerator = np.diag(W_third @ W_third @ W_third)
    denom = degree * (degree - 1)
    denom[denom == 0] = 1
    clustering = numerator / denom

    return {
        "n_nodes": n,
        "mean_strength": np.mean(strength),
        "mean_degree": np.mean(degree),
        "mean_clustering": np.mean(clustering),
        "density": np.sum(M > 0) / (n * (n - 1)) if n > 1 else 0,
        "weights_upper": M[np.triu_indices(n, k=1)],
        "degree_sequence": np.sort(degree)[::-1],
        "strength_sequence": np.sort(strength)[::-1],
    }


def plot_human_comparison():
    # Load 256-d results
    results_path = RESULTS_DIR / "256_mahalanobis_results_master.xlsx"
    if not results_path.exists():
        print("No 256-d results found – skipping human comparison.")
        return

    xls = pd.ExcelFile(results_path, engine="openpyxl")
    sheets = {s: xls.parse(s) for s in xls.sheet_names}
    human_df = sheets.get("human")

    # ── Compute human network stats ────────────────────────────────────────
    human_stats = {}
    for concept in CONCEPTS:
        try:
            human_stats[concept] = _compute_human_stats(concept)
        except Exception as e:
            print(f"  Warning: could not load human network for {concept}: {e}")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Human vs LLM Feature Network Comparison (256-d Mahalanobis)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # ── A: Metric distance heatmap ─────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 0:2])

    metrics_to_compare = ["clustering", "corefit", "strength", "degree"]
    human_metric_keys = {
        "clustering": "hum_clustering",
        "corefit": "hum_corefit",
        "strength": "hum_strength",
        "degree": "hum_degree",
    }

    if human_df is not None:
        best_thresholds = {}
        for concept in CONCEPTS:
            concept_dists = []
            df_clust = sheets.get("clustering")
            if df_clust is None:
                continue
            thresholds = df_clust["threshold"].values

            for t_idx, t in enumerate(thresholds):
                dist = 0
                count = 0
                for metric in metrics_to_compare:
                    df_m = sheets.get(metric)
                    if df_m is None or concept not in df_m.columns:
                        continue
                    hum_key = human_metric_keys[metric]
                    hum_row = human_df[human_df["concept"] == hum_key]
                    if hum_row.empty or concept not in hum_row.columns:
                        continue
                    llm_val = df_m[concept].iloc[t_idx]
                    hum_val = float(hum_row[concept].iloc[0])
                    if hum_val != 0:
                        dist += ((llm_val - hum_val) / abs(hum_val)) ** 2
                    count += 1
                concept_dists.append(np.sqrt(dist / max(count, 1)))

            best_thresholds[concept] = thresholds[np.argmin(concept_dists)]

            # Build heatmap matrix
            if concept == CONCEPTS[0]:
                dist_matrix = np.zeros((len(CONCEPTS), len(thresholds)))
            ci = CONCEPTS.index(concept)
            dist_matrix[ci, :len(concept_dists)] = concept_dists

        im = ax_heat.imshow(dist_matrix, aspect="auto", cmap="YlOrRd")
        ax_heat.set_yticks(range(len(CONCEPTS)))
        ax_heat.set_yticklabels(CONCEPTS, fontsize=8)
        ax_heat.set_xticks(range(len(thresholds)))
        ax_heat.set_xticklabels([f"{t:.2f}" for t in thresholds], fontsize=7, rotation=45)
        ax_heat.set_xlabel("Threshold")
        ax_heat.set_title("A. Normalized Distance: LLM → Human (per concept × threshold)",
                          fontweight="bold")
        plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, label="NRMSE")

        for ci, concept in enumerate(CONCEPTS):
            bt = best_thresholds.get(concept)
            if bt is not None:
                ti = list(thresholds).index(bt) if bt in thresholds else -1
                if ti >= 0:
                    ax_heat.plot(ti, ci, "k*", markersize=10)

    # ── B: Best-match threshold bar chart ──────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 2])
    if best_thresholds:
        y_pos = range(len(CONCEPTS))
        bars = [best_thresholds.get(c, 0) for c in CONCEPTS]
        colors = [CONCEPT_COLORS[c] for c in CONCEPTS]
        ax_bar.barh(y_pos, bars, color=colors, edgecolor="white", linewidth=0.5)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(CONCEPTS, fontsize=8)
        ax_bar.set_xlabel("Best-match threshold")
        ax_bar.set_title("B. Threshold Closest to Human", fontweight="bold")
        ax_bar.invert_yaxis()

    # ── C: Weight distributions ────────────────────────────────────────────
    ax_wdist = fig.add_subplot(gs[1, 0])
    sample_concepts = ["banana", "car", "key", "phone"]
    for i, concept in enumerate(sample_concepts):
        if concept in human_stats:
            w = human_stats[concept]["weights_upper"]
            w_pos = w[w > 0]
            if len(w_pos) > 0:
                ax_wdist.hist(
                    w_pos, bins=40, alpha=0.45,
                    color=CONCEPT_COLORS[concept], density=True,
                    label=f"{concept} (human)", histtype="stepfilled",
                )
    # Pairwise KS-tests on weight distributions
    weight_pools = {}
    for concept in sample_concepts:
        if concept in human_stats:
            w = human_stats[concept]["weights_upper"]
            weight_pools[concept] = w[w > 0]
    ks_lines = []
    sc_list = [c for c in sample_concepts if c in weight_pools]
    for i in range(len(sc_list)):
        for j in range(i + 1, len(sc_list)):
            stat, p = sp_stats.ks_2samp(weight_pools[sc_list[i]], weight_pools[sc_list[j]])
            ks_lines.append(f"{sc_list[i]}–{sc_list[j]}: p={p:.2e}")
    if ks_lines:
        ax_wdist.text(0.98, 0.98, "KS tests:\n" + "\n".join(ks_lines),
                      transform=ax_wdist.transAxes, fontsize=6, ha="right",
                      verticalalignment="top", family="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax_wdist.set_xlabel("Edge weight")
    ax_wdist.set_ylabel("Density")
    ax_wdist.set_title("C. Human Network Weight Distributions", fontweight="bold")
    ax_wdist.legend(fontsize=7)

    # ── D: Degree CDFs ─────────────────────────────────────────────────────
    ax_cdf = fig.add_subplot(gs[1, 1])
    for concept in sample_concepts:
        if concept in human_stats:
            deg = human_stats[concept]["degree_sequence"]
            cdf = np.arange(1, len(deg) + 1) / len(deg)
            ax_cdf.plot(
                np.sort(deg), cdf,
                color=CONCEPT_COLORS[concept], linewidth=1.5,
                label=f"{concept} (n={len(deg)})",
            )
    # Pairwise KS-tests on degree distributions
    deg_pools = {}
    for concept in sample_concepts:
        if concept in human_stats:
            deg_pools[concept] = human_stats[concept]["degree_sequence"]
    ks_lines = []
    sc_list2 = [c for c in sample_concepts if c in deg_pools]
    for i in range(len(sc_list2)):
        for j in range(i + 1, len(sc_list2)):
            stat, p = sp_stats.ks_2samp(deg_pools[sc_list2[i]], deg_pools[sc_list2[j]])
            ks_lines.append(f"{sc_list2[i]}–{sc_list2[j]}: p={p:.2e}")
    if ks_lines:
        ax_cdf.text(0.02, 0.98, "KS tests:\n" + "\n".join(ks_lines),
                    transform=ax_cdf.transAxes, fontsize=6,
                    verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax_cdf.set_xlabel("Degree")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("D. Human Network Degree CDFs", fontweight="bold")
    ax_cdf.legend(fontsize=7)

    # ── E: Network size comparison ─────────────────────────────────────────
    ax_size = fig.add_subplot(gs[1, 2])
    human_sizes = [human_stats[c]["n_nodes"] if c in human_stats else 0 for c in CONCEPTS]
    x = np.arange(len(CONCEPTS))
    ax_size.bar(x, human_sizes, color=[CONCEPT_COLORS[c] for c in CONCEPTS],
                edgecolor="white", linewidth=0.5)
    ax_size.axhline(256, color="gray", linestyle="--", linewidth=1, label="256-d LLM")
    ax_size.set_xticks(x)
    ax_size.set_xticklabels(CONCEPTS, rotation=45, ha="right", fontsize=7)
    ax_size.set_ylabel("Number of nodes / features")
    ax_size.set_title("E. Network Size: Human vs LLM", fontweight="bold")
    ax_size.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "human_vs_llm_comparison.png")


def plot_metric_profiles():
    """Per-metric comparison: human value vs LLM value across thresholds."""
    results_path = RESULTS_DIR / "256_mahalanobis_results_master.xlsx"
    if not results_path.exists():
        return

    xls = pd.ExcelFile(results_path, engine="openpyxl")
    sheets = {s: xls.parse(s) for s in xls.sheet_names}
    human_df = sheets.get("human")
    if human_df is None:
        return

    metrics = {
        "Clustering": ("clustering", "hum_clustering"),
        "Core-Periphery Fit": ("corefit", "hum_corefit"),
        "Diversity": ("diversity18", "hum_diversity"),
        "Modularity": ("modularity18", "hum_modularity"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Human Baseline vs 256-d LLM Metric Trajectories",
        fontsize=14, fontweight="bold", y=0.99,
    )

    for ax, (label, (sheet_key, hum_key)) in zip(axes.flat, metrics.items()):
        df = sheets.get(sheet_key)
        if df is None:
            ax.set_visible(False)
            continue

        thresholds = df["threshold"].values
        hum_row = human_df[human_df["concept"] == hum_key]

        for concept in CONCEPTS:
            if concept not in df.columns:
                continue
            vals = df[concept].values.astype(float)
            ax.plot(thresholds, vals, color=CONCEPT_COLORS[concept],
                    alpha=0.7, linewidth=1.2, label=concept)

            if not hum_row.empty and concept in hum_row.columns:
                hval = float(hum_row[concept].iloc[0])
                ax.axhline(hval, color=CONCEPT_COLORS[concept],
                           linestyle=":", alpha=0.4, linewidth=0.8)

        # Wilcoxon signed-rank: LLM vs human at each threshold
        if not hum_row.empty:
            pvals = []
            for t_idx in range(len(thresholds)):
                llm_vals, hum_vals = [], []
                for concept in CONCEPTS:
                    if concept in df.columns and concept in hum_row.columns:
                        llm_vals.append(float(df[concept].iloc[t_idx]))
                        hum_vals.append(float(hum_row[concept].iloc[0]))
                if len(llm_vals) >= 5:
                    diffs = np.array(llm_vals) - np.array(hum_vals)
                    if np.any(diffs != 0):
                        _, p = sp_stats.wilcoxon(diffs)
                    else:
                        p = 1.0
                    pvals.append(p)
                else:
                    pvals.append(np.nan)

            # Shade thresholds where p > 0.05 (LLM ≈ human)
            for t_idx, p in enumerate(pvals):
                if not np.isnan(p) and p > 0.05:
                    t = thresholds[t_idx]
                    half = (thresholds[1] - thresholds[0]) / 2 if len(thresholds) > 1 else 0.025
                    ax.axvspan(t - half, t + half, color="green", alpha=0.07)

            # Annotate min p-value threshold
            ns_thresholds = [thresholds[i] for i, p in enumerate(pvals)
                             if not np.isnan(p) and p > 0.05]
            if ns_thresholds:
                ax.text(0.02, 0.02,
                        f"p > .05 at t={', '.join(f'{t:.2f}' for t in ns_thresholds)}",
                        transform=ax.transAxes, fontsize=6.5, color="green",
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
            else:
                ax.text(0.02, 0.02, "p < .05 at all thresholds",
                        transform=ax.transAxes, fontsize=6.5, color="red",
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_xlabel("Threshold")
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    savefig(fig, "human_vs_llm_metric_profiles.png")


if __name__ == "__main__":
    plot_human_comparison()
    plot_metric_profiles()
    print("Done – human comparison figures.")
