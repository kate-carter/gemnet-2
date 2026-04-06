#!/usr/bin/env python3
"""
Figure 1 – BCT metric threshold sweeps with human baselines.

Generates a 6-panel figure (2×3) showing how each network metric evolves
across the threshold range for every concept, with human-network baselines
drawn as horizontal dashed lines.  Panels: diversity, modularity, clustering,
core-periphery fit, degree, strength.

Extensible: when results for additional dimensionalities arrive, add their
paths to RESULTS_DIR and the script will overlay them.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats as sp_stats
from vis_config import (
    apply_style, savefig, CONCEPTS, CONCEPT_COLORS, DIM_COLORS,
    RESULTS_DIR, OUTPUT_DIR,
)

apply_style()

# ── Which metric sheets to use (prefer 18-threshold variants) ──────────────
METRIC_MAP = {
    "Diversity":    ("diversity18",  "diversity"),
    "Modularity":   ("modularity18", "modularity"),
    "Clustering":   ("clustering",   None),
    "Core-Periphery Fit": ("corefit", None),
    "Degree":       ("degree",       None),
    "Strength":     ("strength",     None),
}
HUMAN_METRIC_MAP = {
    "Diversity":    "hum_diversity",
    "Modularity":   "hum_modularity",
    "Clustering":   "hum_clustering",
    "Core-Periphery Fit": "hum_corefit",
    "Degree":       "hum_degree",
    "Strength":     "hum_strength",
}


def _pick_sheet(sheets, primary, fallback):
    if primary in sheets:
        return sheets[primary]
    if fallback and fallback in sheets:
        return sheets[fallback]
    return None


def load_all_dims():
    """Return {dim: {sheet_name: DataFrame}} for every available results file."""
    import glob
    results = {}
    for p in sorted(RESULTS_DIR.glob("*_mahalanobis_results_master.xlsx")):
        dim = int(p.stem.split("_")[0])
        xls = pd.ExcelFile(p, engine="openpyxl")
        results[dim] = {s: xls.parse(s) for s in xls.sheet_names}
    return results


def plot_threshold_sweep():
    all_dims = load_all_dims()
    if not all_dims:
        print("No results files found – skipping threshold sweep.")
        return

    available_dims = sorted(all_dims.keys())
    n_dims = len(available_dims)

    for dim in available_dims:
        sheets = all_dims[dim]
        human_df = sheets.get("human")

        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
        fig.suptitle(
            f"Network Metric Threshold Sweeps – {dim}-d Mahalanobis Embeddings",
            fontsize=14, fontweight="bold", y=0.98,
        )

        for ax, (metric_label, (prim, fb)) in zip(axes.flat, METRIC_MAP.items()):
            df = _pick_sheet(sheets, prim, fb)
            if df is None:
                ax.set_visible(False)
                continue

            thresholds = df["threshold"].values

            for concept in CONCEPTS:
                if concept not in df.columns:
                    continue
                vals = df[concept].values.astype(float)
                ax.plot(
                    thresholds, vals,
                    color=CONCEPT_COLORS[concept], alpha=0.7, linewidth=1.3,
                    label=concept,
                )

            # Human baselines
            hum_key = HUMAN_METRIC_MAP.get(metric_label)
            if human_df is not None and hum_key:
                hum_row = human_df[human_df["concept"] == hum_key]
                if not hum_row.empty:
                    for concept in CONCEPTS:
                        if concept in hum_row.columns:
                            hval = float(hum_row[concept].iloc[0])
                            ax.axhline(
                                hval,
                                color=CONCEPT_COLORS[concept],
                                linestyle="--", alpha=0.35, linewidth=0.9,
                            )

            # Wilcoxon signed-rank: LLM vs human at each threshold
            if human_df is not None and hum_key:
                hum_row2 = human_df[human_df["concept"] == hum_key]
                if not hum_row2.empty:
                    sig_markers_x, sig_markers_y = [], []
                    for t_idx in range(len(thresholds)):
                        llm_v, hum_v = [], []
                        for c in CONCEPTS:
                            if c in df.columns and c in hum_row2.columns:
                                llm_v.append(float(df[c].iloc[t_idx]))
                                hum_v.append(float(hum_row2[c].iloc[0]))
                        if len(llm_v) >= 5:
                            diffs = np.array(llm_v) - np.array(hum_v)
                            if np.any(diffs != 0):
                                _, p = sp_stats.wilcoxon(diffs)
                            else:
                                p = 1.0
                            if p > 0.05:
                                sig_markers_x.append(thresholds[t_idx])
                                sig_markers_y.append(np.mean(llm_v))
                    if sig_markers_x:
                        ax.scatter(sig_markers_x, sig_markers_y,
                                   marker="v", color="green", s=30, zorder=10,
                                   alpha=0.7, label="p>.05 vs human" if t_idx == 0 else "")

            ax.set_title(metric_label, fontweight="bold")
            ax.set_xlabel("Threshold")
            ax.set_ylabel(metric_label)
            ax.set_xlim(thresholds.min() - 0.01, thresholds.max() + 0.01)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center", ncol=8, fontsize=7.5, frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        savefig(fig, f"threshold_sweep_{dim}d.png")

    # ── Multi-dimension overlay (if >1 dim available) ──────────────────────
    if n_dims > 1:
        _plot_multidim_overlay(all_dims, available_dims)


def _plot_multidim_overlay(all_dims, available_dims):
    """Per-metric panels with one line per dim (mean across concepts) ± spread."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle(
        "Cross-Dimensional Metric Comparison (mean ± SE across concepts)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for ax, (metric_label, (prim, fb)) in zip(axes.flat, METRIC_MAP.items()):
        for dim in available_dims:
            sheets = all_dims[dim]
            df = _pick_sheet(sheets, prim, fb)
            if df is None:
                continue
            thresholds = df["threshold"].values
            vals = df[CONCEPTS].values.astype(float)
            mean = np.nanmean(vals, axis=1)
            se = np.nanstd(vals, axis=1) / np.sqrt(vals.shape[1])
            color = DIM_COLORS.get(dim, "#333333")
            ax.plot(thresholds, mean, color=color, linewidth=2, label=f"{dim}-d")
            ax.fill_between(thresholds, mean - se, mean + se, color=color, alpha=0.15)

        ax.set_title(metric_label, fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(f"Mean {metric_label}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(available_dims), fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    savefig(fig, "threshold_sweep_multidim.png")


if __name__ == "__main__":
    plot_threshold_sweep()
    print("Done – threshold sweep figures.")
