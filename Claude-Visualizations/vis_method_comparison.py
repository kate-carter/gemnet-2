#!/usr/bin/env python3
"""
Cross-method comparison: which correlation method × dimensionality × threshold
combination best matches human concept networks and best predicts SemD?

Generates three multi-panel figures:
  1. method_comparison_human.png    – human-distance optimization landscape
  2. method_comparison_semd.png     – SemD-prediction optimization landscape
  3. method_comparison_primary.png  – focused on modularity & corefit (primary metrics)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as sp_stats
from vis_config import (
    apply_style, savefig, CONCEPTS, CONCEPT_COLORS, DIM_COLORS,
    BASE, DIMS, OUTPUT_DIR, SEMD_PATH, METHODS,
)

apply_style()

METHOD_ORDER = list(METHODS.keys())

BCT_METRICS = {
    "clustering": "hum_clustering",
    "corefit":    "hum_corefit",
    "strength":   "hum_strength",
    "degree":     "hum_degree",
    "modularity": "hum_modularity",
    "diversity":  "hum_diversity",
}

METRIC_WEIGHTS = {
    "modularity": 2.0,
    "corefit":    2.0,
    "clustering": 1.0,
    "strength":   1.0,
    "degree":     1.0,
    "diversity":  1.0,
}

PRIMARY_METRICS = ["modularity", "corefit"]


# ── Data loading ───────────────────────────────────────────────────────────────

def _find_workbook(results_dir, prefix, dim):
    candidates = [
        results_dir / f"{prefix}_{dim}_results_master.xlsx",
        results_dir / f"{prefix}_{dim}_master_results.xlsx",
        results_dir / f"{dim}_results_master.xlsx",
        results_dir / f"{dim}_master_results.xlsx",
    ]
    return next((p for p in candidates if p.exists()), None)


def _pick_sheet(sheets, name):
    return sheets.get(f"{name}18", sheets.get(name))


def load_all_method_results():
    """Return {method: {dim: {sheet_name: DataFrame}}}."""
    data = {}
    for method in METHOD_ORDER:
        info = METHODS[method]
        data[method] = {}
        for dim in DIMS:
            path = _find_workbook(info["results_dir"], info["prefix"], dim)
            if path is None:
                continue
            try:
                xls = pd.ExcelFile(path, engine="openpyxl")
                data[method][dim] = {s: xls.parse(s) for s in xls.sheet_names}
            except Exception as e:
                print(f"  Warning: {method} {dim}-d: {e}")
    return data


def load_human_baselines(all_data):
    """Extract human baseline from whichever workbook has a 'human' sheet."""
    for method in METHOD_ORDER:
        for dim in sorted(all_data.get(method, {}).keys()):
            sheets = all_data[method][dim]
            if "human" in sheets:
                return sheets["human"]
    return None


def load_semd():
    df = pd.read_excel(SEMD_PATH, sheet_name="Values", engine="openpyxl", header=1)
    df.columns = ["term", "mean_cos", "SemD", "BNC_wordcount",
                   "BNC_contexts", "BNC_freq", "lg_BNC_freq"]
    semd = df[df["term"].isin(CONCEPTS)].copy().set_index("term")
    for col in semd.columns:
        semd[col] = pd.to_numeric(semd[col], errors="coerce")
    return semd


# ── Computation ────────────────────────────────────────────────────────────────

def compute_human_distances(all_data, human_df):
    """
    Per cell (method, dim, threshold): mean NRMSE to human across concepts
    and BCT metrics.

    Returns
    -------
    distances : dict  {(method, dim): ndarray of shape (n_thresholds,)}
    concept_distances : dict  {(method, dim): ndarray (n_thresholds, n_concepts)}
    thresholds : ndarray
    """
    distances = {}
    concept_distances = {}
    thresholds = None

    for method in METHOD_ORDER:
        for dim in sorted(all_data.get(method, {}).keys()):
            sheets = all_data[method][dim]

            ref = None
            for mn in BCT_METRICS:
                ref = _pick_sheet(sheets, mn)
                if ref is not None:
                    break
            if ref is None:
                continue

            t = ref["threshold"].values
            if thresholds is None:
                thresholds = t
            n_t = len(t)

            per_concept = np.full((n_t, len(CONCEPTS)), np.nan)

            for t_idx in range(n_t):
                for ci, concept in enumerate(CONCEPTS):
                    w_sq_diffs, w_sum = 0.0, 0.0
                    for mn, hum_key in BCT_METRICS.items():
                        df_m = _pick_sheet(sheets, mn)
                        if df_m is None or concept not in df_m.columns:
                            continue
                        hum_row = human_df[human_df["concept"] == hum_key]
                        if hum_row.empty or concept not in hum_row.columns:
                            continue
                        llm = float(df_m[concept].iloc[t_idx])
                        hum = float(hum_row[concept].iloc[0])
                        if hum != 0:
                            w = METRIC_WEIGHTS.get(mn, 1.0)
                            w_sq_diffs += w * ((llm - hum) / abs(hum)) ** 2
                            w_sum += w
                    if w_sum > 0:
                        per_concept[t_idx, ci] = np.sqrt(w_sq_diffs / w_sum)

            distances[(method, dim)] = np.nanmean(per_concept, axis=1)
            concept_distances[(method, dim)] = per_concept

    return distances, concept_distances, thresholds


def compute_semd_correlations(all_data, semd):
    """
    Per cell (method, dim, threshold): best Spearman |ρ(SemD, metric)|
    across BCT metrics.

    Returns
    -------
    rho_grid : dict {(method, dim): ndarray (n_thresholds,)}  – max |ρ|
    p_grid   : dict {(method, dim): ndarray (n_thresholds,)}  – p for that ρ
    metric_grid : dict {(method, dim): list[str]}  – winning metric name
    all_rhos : dict {(method, dim): dict {metric_name: ndarray}} – full detail
    thresholds : ndarray
    """
    rho_grid, p_grid, metric_grid, all_rhos = {}, {}, {}, {}

    for method in METHOD_ORDER:
        for dim in sorted(all_data.get(method, {}).keys()):
            sheets = all_data[method][dim]

            ref = None
            for mn in BCT_METRICS:
                ref = _pick_sheet(sheets, mn)
                if ref is not None:
                    break
            if ref is None:
                continue

            t = ref["threshold"].values
            n_t = len(t)

            best_rho = np.full(n_t, np.nan)
            best_p = np.full(n_t, np.nan)
            best_m = [""] * n_t
            metric_rhos = {}

            for mn in BCT_METRICS:
                df_m = _pick_sheet(sheets, mn)
                if df_m is None:
                    continue
                rhos = np.full(n_t, np.nan)
                for t_idx in range(n_t):
                    vals, sd = [], []
                    for concept in CONCEPTS:
                        if concept in df_m.columns and concept in semd.index:
                            v = float(df_m[concept].iloc[t_idx])
                            if not np.isnan(v) and v != 0:
                                vals.append(v)
                                sd.append(float(semd.loc[concept, "SemD"]))
                    if len(vals) >= 5:
                        r, p = sp_stats.spearmanr(sd, vals)
                        rhos[t_idx] = r
                        if abs(r) > abs(best_rho[t_idx]) or np.isnan(best_rho[t_idx]):
                            best_rho[t_idx] = r
                            best_p[t_idx] = p
                            best_m[t_idx] = mn
                metric_rhos[mn] = rhos

            rho_grid[(method, dim)] = np.abs(best_rho)
            p_grid[(method, dim)] = best_p
            metric_grid[(method, dim)] = best_m
            all_rhos[(method, dim)] = metric_rhos

    return rho_grid, p_grid, metric_grid, all_rhos


# ── Helpers for grid layout ────────────────────────────────────────────────────

def _build_row_keys(all_keys):
    """Order row keys as (method, dim) grouped by method, dim ascending."""
    rows = []
    for method in METHOD_ORDER:
        dims_for = sorted(d for m, d in all_keys if m == method)
        for d in dims_for:
            rows.append((method, d))
    return rows


def _build_heatmap_matrix(grid, row_keys, n_thresholds):
    mat = np.full((len(row_keys), n_thresholds), np.nan)
    for ri, key in enumerate(row_keys):
        if key in grid:
            vals = grid[key]
            mat[ri, :len(vals)] = vals
    return mat


def _method_boundaries(row_keys):
    """Return y-positions where method group changes (for dividers)."""
    boundaries = []
    prev = None
    for i, (m, _) in enumerate(row_keys):
        if prev is not None and m != prev:
            boundaries.append(i - 0.5)
        prev = m
    return boundaries


# ── Figure 1: Human distance landscape ─────────────────────────────────────────

def plot_human_comparison_landscape(all_data, human_df):
    if human_df is None:
        print("No human baselines found – skipping cross-method human comparison.")
        return

    distances, concept_dists, thresholds = compute_human_distances(all_data, human_df)
    if not distances or thresholds is None:
        print("No method results available – skipping human comparison landscape.")
        return

    row_keys = _build_row_keys(distances.keys())
    n_t = len(thresholds)
    mat = _build_heatmap_matrix(distances, row_keys, n_t)
    row_labels = [f"{m} {d}-d" for m, d in row_keys]

    # Global best
    valid = np.where(~np.isnan(mat))
    best_ri, best_ci = valid[0][np.argmin(mat[valid])], valid[1][np.argmin(mat[valid])]
    best_method, best_dim = row_keys[best_ri]
    best_thresh = thresholds[best_ci]
    best_val = mat[best_ri, best_ci]

    fig = plt.figure(figsize=(20, 13))
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35,
                           height_ratios=[1.2, 1])
    fig.suptitle(
        "Cross-Method Optimization: LLM → Human Network Distance",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # ── A: Main heatmap ────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0:2])
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("0.92")
    masked = np.ma.masked_invalid(mat)
    im = ax_a.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
    ax_a.set_yticks(range(len(row_labels)))
    ax_a.set_yticklabels(row_labels, fontsize=8)
    ax_a.set_xticks(range(n_t))
    ax_a.set_xticklabels([f"{t:.2f}" for t in thresholds], fontsize=7, rotation=45)
    ax_a.set_xlabel("Threshold")
    ax_a.plot(best_ci, best_ri, "k*", markersize=16, zorder=10)

    for b in _method_boundaries(row_keys):
        ax_a.axhline(b, color="white", linewidth=2)

    plt.colorbar(im, ax=ax_a, fraction=0.025, pad=0.02, label="Mean NRMSE")
    ax_a.set_title("A. Weighted NRMSE to Human (2x mod+corefit; lower = more human-like)",
                    fontweight="bold", fontsize=10)
    ax_a.text(
        0.01, 0.01,
        f"★ Best: {best_method} {best_dim}-d @ t={best_thresh:.2f} (NRMSE={best_val:.4f})",
        transform=ax_a.transAxes, fontsize=8, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.9),
    )

    # ── B: Per-method boxplot at each method's best condition ──────────────
    ax_b = fig.add_subplot(gs[0, 2])
    method_concept_dists = {}
    for method in METHOD_ORDER:
        method_keys = [(m, d) for m, d in distances if m == method]
        if not method_keys:
            continue
        # Find best (dim, threshold) for this method
        best_d_key = min(method_keys, key=lambda k: np.nanmin(distances[k]))
        best_t_idx = int(np.nanargmin(distances[best_d_key]))
        per_c = concept_dists[best_d_key][best_t_idx, :]
        method_concept_dists[method] = per_c[~np.isnan(per_c)]

    active_methods = [m for m in METHOD_ORDER if m in method_concept_dists]
    if len(active_methods) >= 2:
        bp_data = [method_concept_dists[m] for m in active_methods]
        bp = ax_b.boxplot(bp_data, patch_artist=True, widths=0.55)
        for patch, m in zip(bp["boxes"], active_methods):
            patch.set_facecolor(METHODS[m]["color"])
            patch.set_alpha(0.6)
        for i, m in enumerate(active_methods):
            jx = np.random.default_rng(42).normal(i + 1, 0.06, len(bp_data[i]))
            ax_b.scatter(jx, bp_data[i], color=METHODS[m]["color"],
                         s=18, alpha=0.7, edgecolors="white", linewidths=0.3, zorder=5)

        ax_b.set_xticklabels(active_methods, fontsize=8, rotation=30, ha="right")
        ax_b.set_ylabel("Per-concept NRMSE")

        groups = [method_concept_dists[m] for m in active_methods if len(method_concept_dists[m]) >= 2]
        if len(groups) >= 2:
            H, kw_p = sp_stats.kruskal(*groups)
            ax_b.set_title(
                f"B. Method Comparison\n(KW H={H:.1f}, p={kw_p:.3g})",
                fontweight="bold", fontsize=10,
            )

            pw_lines = []
            for i in range(len(active_methods)):
                for j in range(i + 1, len(active_methods)):
                    a = method_concept_dists[active_methods[i]]
                    b = method_concept_dists[active_methods[j]]
                    if len(a) >= 2 and len(b) >= 2:
                        U, mw_p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                        pw_lines.append(
                            f"{active_methods[i][:3]}–{active_methods[j][:3]}: p={mw_p:.2e}"
                        )
            if pw_lines:
                ax_b.text(0.02, 0.98, "\n".join(pw_lines),
                          transform=ax_b.transAxes, fontsize=6.5,
                          verticalalignment="top", family="monospace",
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        else:
            ax_b.set_title("B. Method Comparison", fontweight="bold")
    else:
        ax_b.set_title("B. Method Comparison (need ≥2 methods)", fontweight="bold")
        ax_b.set_visible(False)

    # ── C: LLM vs Human scatter at global-best condition ──────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    best_sheets = all_data[best_method][best_dim]
    llm_vals_all, hum_vals_all, concept_labels = [], [], []
    for mn, hum_key in BCT_METRICS.items():
        df_m = _pick_sheet(best_sheets, mn)
        if df_m is None:
            continue
        hum_row = human_df[human_df["concept"] == hum_key]
        if hum_row.empty:
            continue
        for concept in CONCEPTS:
            if concept in df_m.columns and concept in hum_row.columns:
                llm = float(df_m[concept].iloc[best_ci])
                hum = float(hum_row[concept].iloc[0])
                llm_vals_all.append(llm)
                hum_vals_all.append(hum)
                concept_labels.append(concept)

    if llm_vals_all:
        llm_arr = np.array(llm_vals_all)
        hum_arr = np.array(hum_vals_all)
        colors = [CONCEPT_COLORS[c] for c in concept_labels]
        ax_c.scatter(hum_arr, llm_arr, c=colors, s=25, alpha=0.7,
                     edgecolors="black", linewidths=0.3)

        # Identity line
        lo = min(hum_arr.min(), llm_arr.min())
        hi = max(hum_arr.max(), llm_arr.max())
        ax_c.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.4, label="y=x")

        r, p = sp_stats.spearmanr(hum_arr, llm_arr)
        ax_c.set_xlabel("Human metric value")
        ax_c.set_ylabel("LLM metric value")
        ax_c.set_title(
            f"C. LLM vs Human @ Best\n({best_method} {best_dim}-d t={best_thresh:.2f},"
            f" ρ={r:.3f}, p={p:.2e})",
            fontweight="bold", fontsize=9,
        )
        ax_c.legend(fontsize=7)

    # ── D: Per-metric Wilcoxon at global-best condition ───────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    metric_names, llm_means, hum_means, wilcox_ps = [], [], [], []
    for mn, hum_key in BCT_METRICS.items():
        df_m = _pick_sheet(best_sheets, mn)
        if df_m is None:
            continue
        hum_row = human_df[human_df["concept"] == hum_key]
        if hum_row.empty:
            continue
        llm_v, hum_v = [], []
        for concept in CONCEPTS:
            if concept in df_m.columns and concept in hum_row.columns:
                llm_v.append(float(df_m[concept].iloc[best_ci]))
                hum_v.append(float(hum_row[concept].iloc[0]))
        if len(llm_v) >= 5:
            metric_names.append(mn.capitalize())
            llm_means.append(np.mean(llm_v))
            hum_means.append(np.mean(hum_v))
            diffs = np.array(llm_v) - np.array(hum_v)
            if np.any(diffs != 0):
                _, wp = sp_stats.wilcoxon(diffs)
            else:
                wp = 1.0
            wilcox_ps.append(wp)

    if metric_names:
        x = np.arange(len(metric_names))
        w = 0.35
        ax_d.bar(x - w / 2, hum_means, w, label="Human", color="#888888",
                 edgecolor="white", linewidth=0.5)
        ax_d.bar(x + w / 2, llm_means, w, label="LLM",
                 color=METHODS[best_method]["color"],
                 edgecolor="white", linewidth=0.5)
        for i, wp in enumerate(wilcox_ps):
            sig = "***" if wp < 0.001 else "**" if wp < 0.01 else "*" if wp < 0.05 else "ns"
            y_top = max(llm_means[i], hum_means[i]) * 1.05
            ax_d.text(i, y_top, f"{sig}\np={wp:.2e}", ha="center", fontsize=6,
                      color="green" if wp > 0.05 else "red")
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(metric_names, fontsize=8, rotation=30, ha="right")
        ax_d.set_ylabel("Mean metric value")
        ax_d.legend(fontsize=8)
        ax_d.set_title("D. Per-Metric: LLM vs Human (Wilcoxon)", fontweight="bold")

    # ── E: Top-10 leaderboard ─────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis("off")
    all_cells = []
    for key, dist_arr in distances.items():
        method, dim = key
        for ti, d in enumerate(dist_arr):
            if not np.isnan(d):
                all_cells.append((d, method, dim, thresholds[ti]))
    all_cells.sort()

    header = f"{'Rank':<5} {'Method':<13} {'Dim':>5} {'Thresh':>7} {'NRMSE':>8}"
    lines = [header, "─" * len(header)]
    for rank, (d, m, dim, t) in enumerate(all_cells[:10], 1):
        lines.append(f"{rank:<5} {m:<13} {dim:>5} {t:>7.2f} {d:>8.4f}")

    ax_e.text(0.05, 0.95, "\n".join(lines), transform=ax_e.transAxes,
              fontsize=8, family="monospace", verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="gray"))
    ax_e.set_title("E. Top-10 Conditions (closest to human)", fontweight="bold")

    savefig(fig, "method_comparison_human.png")


# ── Figure 2: SemD prediction landscape ────────────────────────────────────────

def plot_semd_prediction_landscape(all_data, semd):
    rho_grid, p_grid, metric_grid, all_rhos = compute_semd_correlations(all_data, semd)
    if not rho_grid:
        print("No method results available – skipping SemD prediction landscape.")
        return

    row_keys = _build_row_keys(rho_grid.keys())
    thresholds = None
    for key in rho_grid:
        if key in all_rhos:
            for mn, arr in all_rhos[key].items():
                thresholds = None
                sheets = all_data[key[0]][key[1]]
                ref = _pick_sheet(sheets, mn)
                if ref is not None:
                    thresholds = ref["threshold"].values
                    break
        if thresholds is not None:
            break
    if thresholds is None:
        print("Could not determine thresholds – skipping SemD landscape.")
        return

    n_t = len(thresholds)
    mat_rho = _build_heatmap_matrix(rho_grid, row_keys, n_t)
    mat_p = _build_heatmap_matrix(p_grid, row_keys, n_t)
    row_labels = [f"{m} {d}-d" for m, d in row_keys]

    # Global best
    valid = np.where(~np.isnan(mat_rho))
    best_ri = valid[0][np.argmax(mat_rho[valid])]
    best_ci = valid[1][np.argmax(mat_rho[valid])]
    best_method, best_dim = row_keys[best_ri]
    best_thresh = thresholds[best_ci]
    best_rho_val = mat_rho[best_ri, best_ci]
    best_p_val = mat_p[best_ri, best_ci]
    best_metric_name = metric_grid[(best_method, best_dim)][best_ci]

    fig = plt.figure(figsize=(20, 13))
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35,
                           height_ratios=[1.2, 1])
    fig.suptitle(
        "Cross-Method Optimization: Best Predictor of Semantic Diversity (SemD)",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # ── A: Main heatmap of max |ρ| ────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0:2])
    cmap_rho = LinearSegmentedColormap.from_list(
        "rho", ["#f7f7f7", "#fee08b", "#d73027"]
    )
    cmap_rho.set_bad("0.92")
    masked = np.ma.masked_invalid(mat_rho)
    im = ax_a.imshow(masked, aspect="auto", cmap=cmap_rho, vmin=0, vmax=1,
                     interpolation="nearest")

    # Hatch cells where p < .05
    for ri in range(mat_p.shape[0]):
        for ci in range(mat_p.shape[1]):
            if not np.isnan(mat_p[ri, ci]) and mat_p[ri, ci] < 0.05:
                ax_a.add_patch(plt.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=0.3,
                    linestyle="-", hatch="//", alpha=0.25,
                ))

    ax_a.set_yticks(range(len(row_labels)))
    ax_a.set_yticklabels(row_labels, fontsize=8)
    ax_a.set_xticks(range(n_t))
    ax_a.set_xticklabels([f"{t:.2f}" for t in thresholds], fontsize=7, rotation=45)
    ax_a.set_xlabel("Threshold")
    ax_a.plot(best_ci, best_ri, "k*", markersize=16, zorder=10)

    for b in _method_boundaries(row_keys):
        ax_a.axhline(b, color="white", linewidth=2)

    plt.colorbar(im, ax=ax_a, fraction=0.025, pad=0.02, label="max |ρ(SemD, metric)|")
    ax_a.set_title("A. SemD Prediction Strength (higher = better; hatched = p < .05)",
                    fontweight="bold")
    ax_a.text(
        0.01, 0.01,
        f"★ Best: {best_method} {best_dim}-d @ t={best_thresh:.2f}"
        f" — {best_metric_name} |ρ|={best_rho_val:.3f}, p={best_p_val:.3g}",
        transform=ax_a.transAxes, fontsize=8, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.9),
    )

    # ── B: Best |ρ| per method (bar chart) ────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    method_bests = {}
    for method in METHOD_ORDER:
        method_keys = [(m, d) for m, d in rho_grid if m == method]
        if not method_keys:
            continue
        best_key = max(method_keys, key=lambda k: np.nanmax(rho_grid[k]))
        best_ti = int(np.nanargmax(rho_grid[best_key]))
        method_bests[method] = {
            "rho": rho_grid[best_key][best_ti],
            "p": p_grid[best_key][best_ti],
            "dim": best_key[1],
            "thresh": thresholds[best_ti],
            "metric": metric_grid[best_key][best_ti],
        }

    active_methods = [m for m in METHOD_ORDER if m in method_bests]
    if active_methods:
        x = np.arange(len(active_methods))
        rhos = [method_bests[m]["rho"] for m in active_methods]
        colors = [METHODS[m]["color"] for m in active_methods]
        bars = ax_b.bar(x, rhos, color=colors, edgecolor="white", linewidth=0.5)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(active_methods, fontsize=8, rotation=30, ha="right")
        ax_b.set_ylabel("Best |ρ(SemD, metric)|")
        ax_b.set_ylim(0, 1)

        for i, m in enumerate(active_methods):
            info = method_bests[m]
            sig = "***" if info["p"] < 0.001 else "**" if info["p"] < 0.01 \
                  else "*" if info["p"] < 0.05 else "ns"
            ax_b.text(i, info["rho"] + 0.02,
                      f"{sig}\n{info['metric']}\n{info['dim']}-d",
                      ha="center", fontsize=6.5)
        ax_b.set_title("B. Best |ρ| per Method", fontweight="bold")

    # ── C: SemD vs winning metric scatter at global best ──────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    best_sheets = all_data[best_method][best_dim]
    df_m = _pick_sheet(best_sheets, best_metric_name)
    if df_m is not None:
        vals, sd_vals, c_labels = [], [], []
        for concept in CONCEPTS:
            if concept in df_m.columns and concept in semd.index:
                v = float(df_m[concept].iloc[best_ci])
                if not np.isnan(v) and v != 0:
                    vals.append(v)
                    sd_vals.append(float(semd.loc[concept, "SemD"]))
                    c_labels.append(concept)
        if vals:
            colors = [CONCEPT_COLORS[c] for c in c_labels]
            ax_c.scatter(sd_vals, vals, c=colors, s=55, edgecolors="black",
                         linewidths=0.5, zorder=5)
            for i, c in enumerate(c_labels):
                ax_c.annotate(c, (sd_vals[i], vals[i]), fontsize=6.5,
                              xytext=(3, 3), textcoords="offset points")

            r, p = sp_stats.spearmanr(sd_vals, vals)
            ax_c.set_xlabel("SemD")
            ax_c.set_ylabel(f"{best_metric_name.capitalize()}")
            ax_c.set_title(
                f"C. SemD vs {best_metric_name.capitalize()} @ Best\n"
                f"({best_method} {best_dim}-d t={best_thresh:.2f}, ρ={r:.3f}, p={p:.3g})",
                fontweight="bold", fontsize=9,
            )

    # ── D: Per-metric ρ trajectories at best method/dim ───────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    detail_key = (best_method, best_dim)
    if detail_key in all_rhos:
        detail_sheets = all_data[best_method][best_dim]
        ref = None
        for mn in BCT_METRICS:
            ref = _pick_sheet(detail_sheets, mn)
            if ref is not None:
                break
        if ref is not None:
            t_arr = ref["threshold"].values
            for mn, rho_arr in all_rhos[detail_key].items():
                ax_d.plot(t_arr, rho_arr, linewidth=1.5, marker=".", markersize=4,
                          label=mn.capitalize())
            ax_d.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax_d.set_xlabel("Threshold")
            ax_d.set_ylabel("Spearman ρ (SemD × metric)")
            ax_d.set_ylim(-1, 1)
            ax_d.legend(fontsize=7)
            ax_d.set_title(
                f"D. SemD–Metric ρ Profiles\n({best_method} {best_dim}-d)",
                fontweight="bold", fontsize=10,
            )

    # ── E: Top-10 leaderboard ─────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis("off")
    all_cells = []
    for key in rho_grid:
        method, dim = key
        for ti, rho_val in enumerate(rho_grid[key]):
            if not np.isnan(rho_val):
                all_cells.append((
                    rho_val, p_grid[key][ti], metric_grid[key][ti],
                    method, dim, thresholds[ti],
                ))
    all_cells.sort(key=lambda x: -x[0])

    header = f"{'#':<3} {'Method':<11} {'Dim':>5} {'t':>5} {'|ρ|':>6} {'p':>10} {'Metric':<12}"
    lines = [header, "─" * len(header)]
    for rank, (rho_val, pv, mn, m, dim, t) in enumerate(all_cells[:10], 1):
        sig = "*" if pv < 0.05 else " "
        lines.append(
            f"{rank:<3} {m:<11} {dim:>5} {t:>5.2f} {rho_val:>6.3f} {pv:>10.3g}{sig} {mn:<12}"
        )

    ax_e.text(0.02, 0.95, "\n".join(lines), transform=ax_e.transAxes,
              fontsize=7.5, family="monospace", verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="gray"))
    ax_e.set_title("E. Top-10 Conditions (strongest SemD predictor)", fontweight="bold")

    savefig(fig, "method_comparison_semd.png")


# ── Figure 3: Primary metrics – modularity & corefit ───────────────────────────

def _single_metric_rho_grid(all_data, semd, metric_name):
    """Compute ρ(SemD, metric) for a single BCT metric across all cells."""
    rho_grid, p_grid = {}, {}
    for method in METHOD_ORDER:
        for dim in sorted(all_data.get(method, {}).keys()):
            sheets = all_data[method][dim]
            df_m = _pick_sheet(sheets, metric_name)
            if df_m is None:
                continue
            t = df_m["threshold"].values
            n_t = len(t)
            rhos = np.full(n_t, np.nan)
            ps = np.full(n_t, np.nan)
            for t_idx in range(n_t):
                vals, sd = [], []
                for concept in CONCEPTS:
                    if concept in df_m.columns and concept in semd.index:
                        v = float(df_m[concept].iloc[t_idx])
                        if not np.isnan(v) and v != 0:
                            vals.append(v)
                            sd.append(float(semd.loc[concept, "SemD"]))
                if len(vals) >= 5:
                    r, p = sp_stats.spearmanr(sd, vals)
                    rhos[t_idx] = r
                    ps[t_idx] = p
            rho_grid[(method, dim)] = rhos
            p_grid[(method, dim)] = ps
    return rho_grid, p_grid


def plot_primary_metrics(all_data, semd):
    """Focused figure on modularity and corefit — the two primary SemD predictors."""
    mod_rho, mod_p = _single_metric_rho_grid(all_data, semd, "modularity")
    cf_rho, cf_p = _single_metric_rho_grid(all_data, semd, "corefit")

    if not mod_rho and not cf_rho:
        print("No modularity/corefit data – skipping primary metrics figure.")
        return

    all_keys = set(mod_rho.keys()) | set(cf_rho.keys())
    row_keys = _build_row_keys(all_keys)
    row_labels = [f"{m} {d}-d" for m, d in row_keys]

    thresholds = None
    for key in (list(mod_rho.keys()) + list(cf_rho.keys())):
        sheets = all_data[key[0]][key[1]]
        ref = _pick_sheet(sheets, "modularity") or _pick_sheet(sheets, "corefit")
        if ref is not None:
            thresholds = ref["threshold"].values
            break
    if thresholds is None:
        return
    n_t = len(thresholds)

    abs_mod = {k: np.abs(v) for k, v in mod_rho.items()}
    abs_cf = {k: np.abs(v) for k, v in cf_rho.items()}
    mat_mod = _build_heatmap_matrix(abs_mod, row_keys, n_t)
    mat_cf = _build_heatmap_matrix(abs_cf, row_keys, n_t)
    mat_mod_p = _build_heatmap_matrix(mod_p, row_keys, n_t)
    mat_cf_p = _build_heatmap_matrix(cf_p, row_keys, n_t)

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.35,
                           height_ratios=[1.1, 1])
    fig.suptitle(
        "Primary Metrics: Modularity & Core-Periphery Fit as SemD Predictors",
        fontsize=15, fontweight="bold", y=0.99,
    )

    cmap_rho = LinearSegmentedColormap.from_list(
        "rho", ["#f7f7f7", "#fee08b", "#d73027"]
    )
    cmap_rho.set_bad("0.92")

    def _draw_rho_heatmap(ax, mat, mat_p, title, metric_label):
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap=cmap_rho, vmin=0, vmax=1,
                       interpolation="nearest")
        for ri in range(mat_p.shape[0]):
            for ci in range(mat_p.shape[1]):
                if not np.isnan(mat_p[ri, ci]) and mat_p[ri, ci] < 0.05:
                    ax.add_patch(plt.Rectangle(
                        (ci - 0.5, ri - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=0.3,
                        hatch="//", alpha=0.25,
                    ))
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xticks(range(n_t))
        ax.set_xticklabels([f"{t:.2f}" for t in thresholds], fontsize=6, rotation=45)
        ax.set_xlabel("Threshold")
        for b in _method_boundaries(row_keys):
            ax.axhline(b, color="white", linewidth=2)
        valid = np.where(~np.isnan(mat))
        if valid[0].size > 0:
            bi = valid[0][np.argmax(mat[valid])]
            bj = valid[1][np.argmax(mat[valid])]
            ax.plot(bj, bi, "k*", markersize=14, zorder=10)
            bm, bd = row_keys[bi]
            bval = mat[bi, bj]
            bp = mat_p[bi, bj]
            ax.text(0.01, 0.01,
                    f"★ {bm} {bd}-d t={thresholds[bj]:.2f}"
                    f" |ρ|={bval:.3f} p={bp:.3g}",
                    transform=ax.transAxes, fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.9))
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                     label=f"|ρ(SemD, {metric_label})|")
        ax.set_title(title, fontweight="bold", fontsize=10)

    # ── A: Modularity heatmap ──────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_rho_heatmap(ax_a, mat_mod, mat_mod_p,
                      "A. |ρ(SemD, Modularity)|\n(hatched = p < .05)",
                      "modularity")

    # ── B: Corefit heatmap ─────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_rho_heatmap(ax_b, mat_cf, mat_cf_p,
                      "B. |ρ(SemD, Corefit)|\n(hatched = p < .05)",
                      "corefit")

    # ── Identify best cells for scatter panels ─────────────────────────────
    def _best_cell(abs_grid, p_grid):
        best_key, best_ti, best_rho, best_p = None, 0, 0, 1
        for k, arr in abs_grid.items():
            for ti, v in enumerate(arr):
                if not np.isnan(v) and v > best_rho:
                    best_rho = v
                    best_p = p_grid[k][ti]
                    best_key = k
                    best_ti = ti
        return best_key, best_ti, best_rho, best_p

    mod_key, mod_ti, mod_best_rho, mod_best_p = _best_cell(abs_mod, mod_p)
    cf_key, cf_ti, cf_best_rho, cf_best_p = _best_cell(abs_cf, cf_p)

    # ── C: SemD vs modularity scatter at best cell ─────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    if mod_key is not None:
        sheets = all_data[mod_key[0]][mod_key[1]]
        df_m = _pick_sheet(sheets, "modularity")
        if df_m is not None:
            vals, sd_vals, c_labels = [], [], []
            for concept in CONCEPTS:
                if concept in df_m.columns and concept in semd.index:
                    v = float(df_m[concept].iloc[mod_ti])
                    if not np.isnan(v) and v != 0:
                        vals.append(v)
                        sd_vals.append(float(semd.loc[concept, "SemD"]))
                        c_labels.append(concept)
            if vals:
                ax_c.scatter(sd_vals, vals,
                             c=[CONCEPT_COLORS[c] for c in c_labels],
                             s=55, edgecolors="black", linewidths=0.5, zorder=5)
                for i, c in enumerate(c_labels):
                    ax_c.annotate(c, (sd_vals[i], vals[i]), fontsize=6.5,
                                  xytext=(3, 3), textcoords="offset points")
                r, p = sp_stats.spearmanr(sd_vals, vals)
                ax_c.set_xlabel("SemD")
                ax_c.set_ylabel("Modularity")
                ax_c.set_title(
                    f"C. SemD vs Modularity @ Best\n"
                    f"({mod_key[0]} {mod_key[1]}-d t={thresholds[mod_ti]:.2f},"
                    f" ρ={r:.3f}, p={p:.3g})",
                    fontweight="bold", fontsize=9,
                )

    # ── D: SemD vs corefit scatter at best cell ───────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    if cf_key is not None:
        sheets = all_data[cf_key[0]][cf_key[1]]
        df_m = _pick_sheet(sheets, "corefit")
        if df_m is not None:
            vals, sd_vals, c_labels = [], [], []
            for concept in CONCEPTS:
                if concept in df_m.columns and concept in semd.index:
                    v = float(df_m[concept].iloc[cf_ti])
                    if not np.isnan(v) and v != 0:
                        vals.append(v)
                        sd_vals.append(float(semd.loc[concept, "SemD"]))
                        c_labels.append(concept)
            if vals:
                ax_d.scatter(sd_vals, vals,
                             c=[CONCEPT_COLORS[c] for c in c_labels],
                             s=55, edgecolors="black", linewidths=0.5, zorder=5)
                for i, c in enumerate(c_labels):
                    ax_d.annotate(c, (sd_vals[i], vals[i]), fontsize=6.5,
                                  xytext=(3, 3), textcoords="offset points")
                r, p = sp_stats.spearmanr(sd_vals, vals)
                ax_d.set_xlabel("SemD")
                ax_d.set_ylabel("Core-Periphery Fit")
                ax_d.set_title(
                    f"D. SemD vs Corefit @ Best\n"
                    f"({cf_key[0]} {cf_key[1]}-d t={thresholds[cf_ti]:.2f},"
                    f" ρ={r:.3f}, p={p:.3g})",
                    fontweight="bold", fontsize=9,
                )

    # ── E: Modularity ρ trajectories per method ───────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    for method in METHOD_ORDER:
        method_keys = [(m, d) for m, d in mod_rho if m == method]
        if not method_keys:
            continue
        best_key = max(method_keys,
                       key=lambda k: np.nanmax(np.abs(mod_rho[k])))
        rho_arr = mod_rho[best_key]
        sheets = all_data[best_key[0]][best_key[1]]
        ref = _pick_sheet(sheets, "modularity")
        if ref is None:
            continue
        t_arr = ref["threshold"].values
        ax_e.plot(t_arr, rho_arr, linewidth=1.8, marker=".", markersize=5,
                  color=METHODS[method]["color"],
                  label=f"{method} ({best_key[1]}-d)")
        sig_x = [t_arr[i] for i in range(len(t_arr))
                 if not np.isnan(mod_p[best_key][i])
                 and mod_p[best_key][i] < 0.05]
        sig_y = [rho_arr[i] for i in range(len(t_arr))
                 if not np.isnan(mod_p[best_key][i])
                 and mod_p[best_key][i] < 0.05]
        if sig_x:
            ax_e.scatter(sig_x, sig_y, marker="*", s=40, zorder=10,
                         color=METHODS[method]["color"], edgecolors="black",
                         linewidths=0.3)
    ax_e.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_e.set_xlabel("Threshold")
    ax_e.set_ylabel("ρ(SemD, Modularity)")
    ax_e.set_ylim(-1, 1)
    ax_e.legend(fontsize=7)
    ax_e.set_title("E. Modularity–SemD ρ by Method\n(★ = p < .05)",
                    fontweight="bold", fontsize=10)

    # ── F: Corefit ρ trajectories per method ──────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    for method in METHOD_ORDER:
        method_keys = [(m, d) for m, d in cf_rho if m == method]
        if not method_keys:
            continue
        best_key = max(method_keys,
                       key=lambda k: np.nanmax(np.abs(cf_rho[k])))
        rho_arr = cf_rho[best_key]
        sheets = all_data[best_key[0]][best_key[1]]
        ref = _pick_sheet(sheets, "corefit")
        if ref is None:
            continue
        t_arr = ref["threshold"].values
        ax_f.plot(t_arr, rho_arr, linewidth=1.8, marker=".", markersize=5,
                  color=METHODS[method]["color"],
                  label=f"{method} ({best_key[1]}-d)")
        sig_x = [t_arr[i] for i in range(len(t_arr))
                 if not np.isnan(cf_p[best_key][i])
                 and cf_p[best_key][i] < 0.05]
        sig_y = [rho_arr[i] for i in range(len(t_arr))
                 if not np.isnan(cf_p[best_key][i])
                 and cf_p[best_key][i] < 0.05]
        if sig_x:
            ax_f.scatter(sig_x, sig_y, marker="*", s=40, zorder=10,
                         color=METHODS[method]["color"], edgecolors="black",
                         linewidths=0.3)
    ax_f.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_f.set_xlabel("Threshold")
    ax_f.set_ylabel("ρ(SemD, Corefit)")
    ax_f.set_ylim(-1, 1)
    ax_f.legend(fontsize=7)
    ax_f.set_title("F. Corefit–SemD ρ by Method\n(★ = p < .05)",
                    fontweight="bold", fontsize=10)

    savefig(fig, "method_comparison_primary.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def run_all():
    print("Loading results across all methods and dimensionalities...")
    all_data = load_all_method_results()
    available = {m: sorted(all_data[m].keys()) for m in METHOD_ORDER if all_data[m]}
    for m, dims in available.items():
        print(f"  {m}: {dims}")
    if not available:
        print("No results found for any method.")
        return

    human_df = load_human_baselines(all_data)
    if human_df is not None:
        print(f"  Human baselines loaded ({len(human_df)} metrics)")
    else:
        print("  Warning: no human baseline sheet found in any workbook")

    semd = load_semd()
    print(f"  SemD loaded for {len(semd)} concepts")

    print("\nGenerating cross-method human-distance landscape...")
    plot_human_comparison_landscape(all_data, human_df)

    print("Generating cross-method SemD prediction landscape...")
    plot_semd_prediction_landscape(all_data, semd)

    print("Generating primary metrics (modularity & corefit) figure...")
    plot_primary_metrics(all_data, semd)


if __name__ == "__main__":
    run_all()
    print("Done – cross-method comparison figures.")
