"""
vis_human_alignment.py — Correlating Structural Transitions with Human Network Fit
====================================================================================

Adapted from human_alignment.py (test-pipeline) for the gemnet-2 multi-method
pipeline.  One 9-panel figure is generated per correlation method.

Human BCT baselines are loaded from the canonical human_stats.xlsx (PI published
values).  LLM BCT values are aggregated (mean) across all available
dimensionalities.  Percolation curves are computed from 256-d thresholded
matrices (first alphabetically by concept-prefix, which resolves to 256-d).

Panels
------
  A  Cross-concept Spearman ρ (LLM vs human) across thresholds, overlaid with
     the mean percolation rate of change.
  B  Per-concept z-scored distance heatmap for core-fit.
  C  Same as B but for modularity.
  D  Percolation landscape (giant-component fraction heatmap) with best-fit
     threshold symbols for all six BCT metrics overlaid.
  E  Scatter: percolation midpoint vs threshold of best human fit.
  F  Scatter: percolation steepness vs minimum z-scored distance to human.
  F₂ Same as F without modularity, with regression lines.
  G  Multi-metric z-distance bar chart at the threshold of best core-fit.
  H  Best-fit threshold dot plot per concept (threshold on x-axis).
  I  Same as H but x-axis is edge-retention percentage.

Output: vis_results/human_alignment_{Method}.png  (one file per method)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy.stats import spearmanr, mannwhitneyu, linregress
from scipy.optimize import curve_fit

from vis_config import (
    apply_style, savefig, OUTPUT_DIR,
    CONCEPTS, DIMS, METHODS, HUMAN_STATS_PATH,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

METRICS = ["corefit", "clustering", "strength", "degree", "modularity", "diversity"]
HUMAN_ROW_MAP = {m: f"hum_{m}" for m in METRICS}


# ── Human data ────────────────────────────────────────────────────────────────

def load_human_baselines():
    """Load human BCT values from human_stats.xlsx.

    Returns {metric: {concept: float}}.
    """
    if not HUMAN_STATS_PATH.exists():
        print(f"  ERROR: {HUMAN_STATS_PATH} not found.")
        return {}
    try:
        hdf = pd.read_excel(HUMAN_STATS_PATH, sheet_name="human_bct",
                            engine="openpyxl").set_index("concept")
        human = {}
        for metric in METRICS:
            row_key = HUMAN_ROW_MAP[metric]
            if row_key in hdf.index:
                human[metric] = {
                    c: float(hdf.loc[row_key, c])
                    for c in CONCEPTS if c in hdf.columns
                }
        return human
    except Exception as exc:
        print(f"  ERROR loading human_stats.xlsx: {exc}")
        return {}


# ── LLM BCT data ──────────────────────────────────────────────────────────────

def _find_workbook(results_dir, prefix, dim):
    candidates = [
        results_dir / f"{prefix}_{dim}_results_master.xlsx",
        results_dir / f"{prefix}_{dim}_master_results.xlsx",
        results_dir / f"{dim}_results_master.xlsx",
        results_dir / f"{dim}_master_results.xlsx",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_llm_data(results_dir, prefix):
    """Load and average LLM BCT stats across all available dims.

    Returns (llm, thresholds) where:
        llm        = {metric: {threshold: {concept: value}}}
        thresholds = sorted list of float
    """
    # Collect (metric, threshold, concept, value) across all dims
    records = []
    for dim in DIMS:
        path = _find_workbook(results_dir, prefix, dim)
        if path is None:
            continue
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
        except Exception:
            continue
        for sheet in xls.sheet_names:
            metric = sheet.strip().lower()
            if metric not in METRICS:
                continue
            df = xls.parse(sheet)
            if df.empty:
                continue
            if "threshold" not in df.columns:
                df = df.rename(columns={df.columns[0]: "threshold"})
            for _, row in df.iterrows():
                t = float(row["threshold"])
                for c in CONCEPTS:
                    if c in row.index and not pd.isna(row[c]):
                        records.append((metric, t, c, float(row[c])))

    if not records:
        return {}, []

    raw = pd.DataFrame(records, columns=["metric", "threshold", "concept", "value"])

    # Average over dims for each (metric, threshold, concept)
    agg = (raw.groupby(["metric", "threshold", "concept"])["value"]
           .mean().reset_index())

    llm = {}
    for metric in METRICS:
        sub = agg[agg["metric"] == metric]
        if sub.empty:
            continue
        llm[metric] = {}
        for t, grp in sub.groupby("threshold"):
            llm[metric][float(t)] = {
                row["concept"]: row["value"]
                for _, row in grp.iterrows()
            }

    thresholds = sorted({t for m_data in llm.values() for t in m_data})
    return llm, thresholds


# ── Percolation ───────────────────────────────────────────────────────────────

def _load_matrix(mat_dir, prefix, threshold):
    for pat in [
        f"{prefix}_{threshold:.2f}.csv",
        f"{prefix}_{threshold:.1f}.csv",
    ]:
        path = mat_dir / pat
        if path.exists():
            try:
                probe = pd.read_csv(path, nrows=0)
                try:
                    float(str(probe.columns[0]))
                    df = pd.read_csv(path, header=None)
                except (ValueError, IndexError):
                    df = pd.read_csv(path, index_col=0, header=0)
                vals = df.values.copy().astype(np.float64)
                np.fill_diagonal(vals, 0.0)
                vals = np.nan_to_num(vals, nan=0.0)
                vals = np.clip(vals, 0.0, None)
                return vals
            except Exception:
                pass
    return None


def _resolve_prefix(mat_dir, concept):
    """Find the filename prefix for a concept (picks lowest dim alphabetically)."""
    pat = re.compile(r"^(.+?)_([\d]+\.[0-9]+)\.csv$")
    candidates = set()
    for fname in os.listdir(mat_dir):
        m = pat.match(fname)
        if m:
            pfx = m.group(1)
            if pfx == concept or pfx.startswith(concept + "_"):
                candidates.add(pfx)
    if not candidates:
        return concept
    return concept if concept in candidates else sorted(candidates)[0]


def _giant_fraction(adj):
    n = adj.shape[0]
    binary = (adj > 0)
    visited = np.zeros(n, dtype=bool)
    max_size = 0
    for start in range(n):
        if visited[start]:
            continue
        stack, size = [start], 0
        visited[start] = True
        while stack:
            node = stack.pop()
            size += 1
            for nb in np.where(binary[node])[0]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        max_size = max(max_size, size)
    return max_size / n if n > 0 else 0.0


def compute_percolation(mat_dir, concepts, thresholds):
    """Giant-component fraction and edge retention per concept × threshold."""
    thresh_desc = sorted(thresholds, reverse=True)
    prefix_map  = {c: _resolve_prefix(mat_dir, c) for c in concepts}
    perc, retention = {}, {}
    for c in concepts:
        pfx = prefix_map[c]
        gf_vals, ret_vals = [], []
        for t in thresh_desc:
            mat = _load_matrix(mat_dir, pfx, t)
            if mat is not None:
                gf_vals.append(_giant_fraction(mat))
                n = mat.shape[0]
                possible = n * (n - 1) / 2
                n_edges  = int(np.count_nonzero(np.triu(mat, k=1)))
                ret_vals.append(n_edges / possible * 100 if possible > 0 else 0.0)
            else:
                gf_vals.append(np.nan)
                ret_vals.append(np.nan)
        perc[c]      = np.array(gf_vals)
        retention[c] = np.array(ret_vals)
    return perc, retention, thresh_desc


# ── Logistic fit ──────────────────────────────────────────────────────────────

def _logistic(x, L, k, x0, b):
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def fit_percolation(thresh_desc, y_vals):
    x = np.array(thresh_desc, dtype=float)
    y = np.array(y_vals, dtype=float)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return {"midpoint": np.nan, "steepness": np.nan}
    cross = x[np.argmin(np.abs(y - 0.5))] if len(y) else x.mean()
    try:
        popt, _ = curve_fit(
            _logistic, x, y, p0=[1.0, -10.0, cross, 0.0],
            bounds=([0.5, -200, x.min(), -0.5], [1.5, 200, x.max(), 0.5]),
            maxfev=5000)
        return {"midpoint": popt[2], "steepness": abs(popt[1])}
    except (RuntimeError, ValueError):
        return {"midpoint": np.nan, "steepness": np.nan}


# ── Distance matrices (z-scored) ─────────────────────────────────────────────

def compute_distance_matrices(human, llm, thresholds, concepts):
    """Z-score both LLM and human across concepts, return |z_LLM - z_Human|."""
    dist_matrices = {}
    for metric in METRICS:
        if metric not in llm or metric not in human:
            continue
        h_raw = np.array([human[metric].get(c, np.nan) for c in concepts])
        h_std = np.nanstd(h_raw)
        h_std = h_std if h_std > 0 else 1.0
        h_z   = (h_raw - np.nanmean(h_raw)) / h_std

        mat = np.full((len(concepts), len(thresholds)), np.nan)
        h_valid = ~np.isnan(h_raw)
        for j, t in enumerate(thresholds):
            llm_at_t = llm[metric].get(t, {})
            l_raw = np.array([llm_at_t.get(c, np.nan) for c in concepts])
            valid = h_valid & ~np.isnan(l_raw)
            if valid.sum() < 2:
                continue
            l_std = np.nanstd(l_raw[valid])
            l_std = l_std if l_std > 0 else 1.0
            for i in range(len(concepts)):
                if valid[i]:
                    l_z = (l_raw[i] - np.nanmean(l_raw[valid])) / l_std
                    mat[i, j] = abs(l_z - h_z[i])
        dist_matrices[metric] = mat
    return dist_matrices


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_distance_heatmap(ax, metric, dist_matrices, thresholds,
                           concepts, fits, title):
    if metric not in dist_matrices:
        ax.text(0.5, 0.5, f"No {metric} data", transform=ax.transAxes, ha="center")
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        return
    dm = dist_matrices[metric]
    best_thresholds = []
    for i in range(len(concepts)):
        row = dm[i, :]
        best_thresholds.append(thresholds[int(np.nanargmin(row))]
                               if not np.all(np.isnan(row)) else np.nan)

    vmax = np.nanpercentile(dm, 90)
    im = ax.imshow(dm, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=max(vmax, 0.01), interpolation="nearest")

    thresh_list = list(thresholds)
    for i, bt in enumerate(best_thresholds):
        if not np.isnan(bt) and bt in thresh_list:
            ax.plot(thresh_list.index(bt), i, "*", color="white",
                    markersize=10, markeredgecolor="black", markeredgewidth=0.5)
    for i, c in enumerate(concepts):
        mp = fits[c]["midpoint"]
        if not np.isnan(mp):
            j = int(np.argmin([abs(t - mp) for t in thresh_list]))
            ax.plot(j, i, "|", color="cyan", markersize=12, markeredgewidth=2)

    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=9)
    tick_pos = list(range(0, len(thresholds), 2))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{thresh_list[k]:.2f}" for k in tick_pos],
                       fontsize=9, rotation=45)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("|z_LLM − z_Human|", fontsize=9)
    ax.text(0.02, -0.09, "★ = best-fit threshold    | = percolation midpoint",
            transform=ax.transAxes, fontsize=9, color="#555555")


def _plot_percolation_heatmap(ax, dist_matrices, thresholds, concepts,
                              perc, thresh_desc, title):
    n_c, n_t = len(concepts), len(thresholds)
    thresh_list = list(thresholds)
    perc_mat = np.full((n_c, n_t), np.nan)
    for i, c in enumerate(concepts):
        if c not in perc:
            continue
        for j, t in enumerate(thresh_list):
            matches = [k for k, td in enumerate(thresh_desc) if abs(td - t) < 1e-6]
            if matches and matches[0] < len(perc[c]):
                perc_mat[i, j] = perc[c][matches[0]]

    im = ax.imshow(perc_mat, aspect="auto", cmap="YlGnBu",
                   vmin=0, vmax=1, interpolation="nearest")

    metric_symbols = {
        "corefit":    ("★", 12, "white"),
        "modularity": ("◆",  9, "#e41a1c"),
        "clustering": ("▲",  8, "#ff7f0e"),
        "strength":   ("●",  8, "#2ca02c"),
        "degree":     ("■",  8, "#984ea3"),
        "diversity":  ("×", 10, "#17becf"),
    }
    cell_counts = defaultdict(int)
    for i in range(n_c):
        for metric in METRICS:
            if metric not in dist_matrices:
                continue
            row = dist_matrices[metric][i, :]
            if np.all(np.isnan(row)):
                continue
            best_j = int(np.nanargmin(row))
            sym, size, color = metric_symbols[metric]
            k = cell_counts[(i, best_j)]
            y_off = (k - 2.5) * 0.12
            cell_counts[(i, best_j)] += 1
            ax.text(best_j, i + y_off, sym, ha="center", va="center",
                    fontsize=size, color=color, fontweight="bold", zorder=5)

    ax.set_yticks(range(n_c))
    ax.set_yticklabels(concepts, fontsize=9)
    tick_pos = list(range(0, n_t, 2))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{thresh_list[k]:.2f}" for k in tick_pos],
                       fontsize=9, rotation=45)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Giant Component Fraction", fontsize=9)
    legend_parts = "  ".join(
        f"{sym} {m}" for m, (sym, _, _) in metric_symbols.items()
        if m in dist_matrices)
    ax.text(0.02, -0.09, f"Best-fit per metric (z-scored):  {legend_parts}",
            transform=ax.transAxes, fontsize=8, color="#555555")


def _scatter_midpoint_vs_bestfit(ax, dist_matrices, thresholds, concepts, fits):
    all_metric_specs = [
        ("corefit",    "o", "Core-fit",   "#e41a1c"),
        ("modularity", "s", "Modularity", "#377eb8"),
        ("clustering", "^", "Clustering", "#ff7f0e"),
        ("strength",   "d", "Strength",   "#2ca02c"),
        ("degree",     "v", "Degree",     "#984ea3"),
        ("diversity",  "P", "Diversity",  "#17becf"),
    ]
    text_y = 0.98
    thresh_list = list(thresholds)
    for metric, marker, label, color in all_metric_specs:
        if metric not in dist_matrices:
            continue
        dm = dist_matrices[metric]
        mids, best_ts = [], []
        for i, c in enumerate(concepts):
            mp = fits[c]["midpoint"]
            row = dm[i, :]
            if np.isnan(mp) or np.all(np.isnan(row)):
                continue
            mids.append(mp)
            best_ts.append(thresh_list[int(np.nanargmin(row))])
        ax.scatter(mids, best_ts, s=70, c=color, marker=marker,
                   edgecolors="white", linewidths=0.5, alpha=0.85,
                   label=label, zorder=3)
        if len(mids) >= 4:
            r, p = spearmanr(mids, best_ts)
            z = np.polyfit(mids, best_ts, 1)
            x_ln = np.linspace(min(mids), max(mids), 50)
            ax.plot(x_ln, np.polyval(z, x_ln), "--", color=color, alpha=0.4, lw=1.2)
            ax.text(0.02, text_y, f"{label}: ρ={r:.3f}, p={p:.3f}",
                    transform=ax.transAxes, fontsize=8, color=color, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            text_y -= 0.07
    ax.set_xlabel("Percolation Midpoint", fontsize=12)
    ax.set_ylabel("Threshold of Best Human Fit", fontsize=12)
    ax.set_title("E.  Phase Transition vs Best-Fit Threshold",
                 fontsize=13, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.2)
    return all_metric_specs


def _scatter_steepness(ax, all_metric_specs, dist_matrices, thresholds,
                       concepts, fits, title, show_regression=False):
    text_y = 0.98
    thresh_list = list(thresholds)
    for metric, marker, label, color in all_metric_specs:
        if metric not in dist_matrices:
            continue
        dm = dist_matrices[metric]
        steeps, min_dists, clabels = [], [], []
        for i, c in enumerate(concepts):
            st  = fits[c]["steepness"]
            row = dm[i, :]
            if np.isnan(st) or np.all(np.isnan(row)):
                continue
            steeps.append(st)
            min_dists.append(np.nanmin(row))
            clabels.append(c)
        ax.scatter(steeps, min_dists, s=70, c=color, marker=marker,
                   edgecolors="white", linewidths=0.5, alpha=0.85,
                   label=label, zorder=3)
        for x, y, lbl in zip(steeps, min_dists, clabels):
            ax.annotate(lbl, (x, y), fontsize=6, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points",
                        color=color, alpha=0.6)
        if len(steeps) >= 4:
            r_s, p_s = spearmanr(steeps, min_dists)
            if show_regression:
                lr = linregress(steeps, min_dists)
                x_ln = np.linspace(min(steeps), max(steeps), 50)
                ax.plot(x_ln, lr.slope * x_ln + lr.intercept, "--",
                        color=color, alpha=0.5, lw=1.5)
                ax.text(0.02, text_y,
                        f"{label}: ρ={r_s:.3f}, p={p_s:.3f}  |  R²={lr.rvalue**2:.3f}",
                        transform=ax.transAxes, fontsize=8, color=color, va="top",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            else:
                ax.text(0.02, text_y, f"{label}: ρ={r_s:.3f}, p={p_s:.3f}",
                        transform=ax.transAxes, fontsize=8, color=color, va="top",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            text_y -= 0.07
    ax.set_xlabel("Percolation Steepness |k|", fontsize=12)
    ax.set_ylabel("Min Z-scored Distance to Human", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.2)


def _bar_multimetric(ax, dist_matrices, concepts):
    metric_keys   = ["corefit", "clustering", "strength", "degree", "modularity", "diversity"]
    metric_labels = ["Core-fit", "Clustering", "Strength", "Degree", "Modularity", "Diversity"]
    colors = ["#e41a1c", "#ff7f0e", "#2ca02c", "#17becf", "#377eb8", "#984ea3"]
    profiles = []
    if "corefit" not in dist_matrices:
        ax.text(0.5, 0.5, "No corefit data", transform=ax.transAxes, ha="center")
        ax.set_title("G.  Multi-Metric Z-Distance at Best Core-fit Threshold",
                     fontsize=13, fontweight="bold", loc="left")
        return
    dm_cf = dist_matrices["corefit"]
    for i in range(len(concepts)):
        row_cf = dm_cf[i, :]
        if np.all(np.isnan(row_cf)):
            continue
        best_j = int(np.nanargmin(row_cf))
        profiles.append([
            dist_matrices[mk][i, best_j] if mk in dist_matrices else np.nan
            for mk in metric_keys
        ])
    if profiles:
        arr = np.array(profiles)
        ax.bar(range(len(metric_labels)), np.nanmean(arr, axis=0),
               yerr=np.nanstd(arr, axis=0), color=colors, alpha=0.8,
               edgecolor="white", linewidth=0.5, capsize=4)
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, fontsize=11, rotation=15, ha="right")
        ax.axhline(0, color="green", ls="--", alpha=0.4, lw=1.5)
        ax.set_ylim(bottom=0)
    ax.set_ylabel("Z-scored Distance (lower = closer)", fontsize=12)
    ax.set_title("G.  Multi-Metric Z-Distance at Best Core-fit Threshold",
                 fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2, axis="y")


def _dotplot_bestfit(ax, dist_matrices, thresholds, concepts,
                     retention, thresh_desc, mode, title):
    thresh_list = list(thresholds)
    dense_group = {"modularity", "strength", "diversity"}
    mid_group   = {"corefit", "clustering", "degree"}
    metric_style = {
        "corefit":    ("o", "#e41a1c"), "modularity": ("s", "#377eb8"),
        "clustering": ("^", "#ff7f0e"), "strength":   ("d", "#2ca02c"),
        "degree":     ("v", "#984ea3"), "diversity":  ("P", "#17becf"),
    }
    # Build threshold → retention map
    t2ret = {}
    for c in concepts:
        t2ret[c] = {}
        for j, td in enumerate(thresh_desc):
            if c in retention and j < len(retention[c]):
                t2ret[c][td] = retention[c][j]

    def _to_x(c, t_val):
        if mode == "retention":
            return t2ret.get(c, {}).get(t_val, np.nan)
        return t_val

    concept_data = {}
    for i, c in enumerate(concepts):
        best = {}
        for metric in METRICS:
            if metric not in dist_matrices:
                continue
            row = dist_matrices[metric][i, :]
            if np.all(np.isnan(row)):
                continue
            best[metric] = thresh_list[int(np.nanargmin(row))]
        concept_data[c] = best

    for i, c in enumerate(concepts):
        bt = concept_data.get(c, {})
        for metric in METRICS:
            if metric not in bt:
                continue
            x_val = _to_x(c, bt[metric])
            if np.isnan(x_val):
                continue
            mk, col = metric_style[metric]
            ax.scatter(x_val, i, marker=mk, s=40, c=col, alpha=0.35,
                       edgecolors="none", zorder=2)

    all_dense, all_mid = [], []
    for i, c in enumerate(concepts):
        bt = concept_data.get(c, {})
        all_x, dense_x, mid_x = [], [], []
        for metric in METRICS:
            if metric not in bt:
                continue
            x_val = _to_x(c, bt[metric])
            if not np.isnan(x_val):
                all_x.append(x_val)
                if metric in dense_group:
                    dense_x.append(x_val)
                    all_dense.append(x_val)
                elif metric in mid_group:
                    mid_x.append(x_val)
                    all_mid.append(x_val)

        y_off = 0.22
        for data, fmt, col, lbl in [
            (all_x,   "D", "black",   "Overall mean" if i == 0 else ""),
            (dense_x, "<", "#377eb8", "Dense group" if i == 0 else ""),
            (mid_x,   ">", "#e41a1c", "Mid-range group" if i == 0 else ""),
        ]:
            if not data:
                continue
            mn = np.mean(data)
            se = np.std(data, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else 0
            y_pos = i if fmt == "D" else (i - y_off if fmt == "<" else i + y_off)
            ax.errorbar(mn, y_pos, xerr=se, fmt=fmt, color=col,
                        markersize=6 if fmt != "D" else 7,
                        capsize=3, capthick=1, elinewidth=1, zorder=4, label=lbl)

    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=10)
    ax.set_ylim(-0.6, len(concepts) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Edge Retention (%)" if mode == "retention" else "Threshold",
                  fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2, axis="x")

    if len(all_dense) >= 3 and len(all_mid) >= 3:
        stat, p = mannwhitneyu(all_dense, all_mid, alternative="two-sided")
        ax.text(0.02, -0.06, f"Dense vs Mid-range: U={stat:.0f}, p={p:.2e}",
                transform=ax.transAxes, fontsize=9, color="#555555",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    metric_handles = [
        plt.Line2D([0], [0], marker=mk, color="w", markerfacecolor=cl,
                   markersize=7, alpha=0.5, linestyle="None", label=name)
        for name, (mk, cl) in [
            ("Core-fit",   ("o", "#e41a1c")), ("Modularity", ("s", "#377eb8")),
            ("Clustering", ("^", "#ff7f0e")), ("Strength",   ("d", "#2ca02c")),
            ("Degree",     ("v", "#984ea3")), ("Diversity",  ("P", "#17becf")),
        ]
    ]
    all_handles = ax.get_legend_handles_labels()[0] + metric_handles
    ax.legend(handles=all_handles, fontsize=7.5, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), ncol=1, frameon=True,
              fancybox=True, borderaxespad=0)


# ── Main figure builder ───────────────────────────────────────────────────────

def _plot_human_alignment(method_name, cfg):
    results_dir = cfg["results_dir"]
    mat_dir     = cfg["matrices_dir"]
    prefix      = cfg["prefix"]

    print(f"  [{method_name}] Loading human baselines …")
    human = load_human_baselines()
    if not human:
        print(f"    No human data — skipping.")
        return

    print(f"  [{method_name}] Loading LLM BCT data …")
    llm, thresholds = load_llm_data(results_dir, prefix)
    if not llm:
        print(f"    No LLM data found — skipping.")
        return
    print(f"    {len(thresholds)} thresholds, metrics: {list(llm.keys())}")

    print(f"  [{method_name}] Computing percolation curves …")
    perc, retention, thresh_desc = compute_percolation(mat_dir, CONCEPTS, thresholds)

    print(f"  [{method_name}] Fitting logistic models …")
    fits = {c: fit_percolation(thresh_desc, perc[c]) for c in CONCEPTS}

    print(f"  [{method_name}] Computing z-scored distance matrices …")
    dist_matrices = compute_distance_matrices(human, llm, thresholds, CONCEPTS)

    # ── Figure ──────────────────────────────────────────────────────────────
    apply_style()
    fig = plt.figure(figsize=(30, 44))
    gs  = gridspec.GridSpec(5, 2, hspace=0.45, wspace=0.32,
                            left=0.08, right=0.88, top=0.94, bottom=0.04)

    n  = len(CONCEPTS)
    try:
        cmap = plt.colormaps["tab20"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")
    cc = {c: cmap(i / max(n - 1, 1)) for i, c in enumerate(CONCEPTS)}

    # Panel A
    ax_a = fig.add_subplot(gs[0, 0])
    thresh_list = list(thresholds)
    for metric, style, label in [
        ("corefit",    "-o", "Core-fit"),
        ("modularity", "-s", "Modularity"),
        ("clustering", "--^", "Clustering"),
        ("strength",   "--d", "Strength"),
    ]:
        if metric not in llm or metric not in human:
            continue
        rs = []
        for t in thresh_list:
            h_vec = [human[metric].get(c, np.nan) for c in CONCEPTS]
            l_vec = [llm[metric].get(t, {}).get(c, np.nan) for c in CONCEPTS]
            mask  = ~(np.isnan(h_vec) | np.isnan(l_vec))
            if mask.sum() >= 4:
                r, _ = spearmanr(np.array(h_vec)[mask], np.array(l_vec)[mask])
            else:
                r = np.nan
            rs.append(r)
        ax_a.plot(thresh_list, rs, style, markersize=5, linewidth=2,
                  label=f"{label} ρ", alpha=0.85)

    ax_a2 = ax_a.twinx()
    mean_perc = np.nanmean([perc[c] for c in CONCEPTS], axis=0)
    dy = -np.gradient(mean_perc, thresh_desc)
    ax_a2.fill_between(thresh_desc, 0, dy, color="grey", alpha=0.15)
    ax_a2.plot(thresh_desc, dy, "-", color="grey", alpha=0.4, linewidth=1.5,
               label="Mean −dGiant/dθ")
    ax_a2.set_ylabel("Mean percolation rate of change", fontsize=10, color="grey")
    ax_a2.tick_params(axis="y", colors="grey")
    ax_a.set_xlabel("Threshold", fontsize=12)
    ax_a.set_ylabel("Spearman ρ (LLM vs Human)", fontsize=12)
    ax_a.set_title("A.  Human-LLM Correlation Across Thresholds",
                   fontsize=13, fontweight="bold", loc="left")
    ax_a.axhline(0, color="grey", ls="--", alpha=0.3)
    ax_a.grid(True, alpha=0.2)
    lines1, labels1 = ax_a.get_legend_handles_labels()
    lines2, labels2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    # Panels B, C, D
    _plot_distance_heatmap(fig.add_subplot(gs[0, 1]), "corefit", dist_matrices,
                           thresh_list, CONCEPTS, fits,
                           "B.  Core-fit Distance to Human (z-scored)")
    _plot_distance_heatmap(fig.add_subplot(gs[1, 0]), "modularity", dist_matrices,
                           thresh_list, CONCEPTS, fits,
                           "C.  Modularity Distance to Human (z-scored)")
    _plot_percolation_heatmap(fig.add_subplot(gs[1, 1]), dist_matrices,
                              thresh_list, CONCEPTS, perc, thresh_desc,
                              "D.  Percolation Landscape × Best-Fit Thresholds")

    # Panels E, F, F₂
    all_metric_specs = _scatter_midpoint_vs_bestfit(
        fig.add_subplot(gs[2, 0]), dist_matrices, thresh_list, CONCEPTS, fits)
    _scatter_steepness(fig.add_subplot(gs[2, 1]), all_metric_specs,
                       dist_matrices, thresh_list, CONCEPTS, fits,
                       "F.  Transition Steepness vs Best Achievable Fit")
    no_mod = [s for s in all_metric_specs if s[0] != "modularity"]
    _scatter_steepness(fig.add_subplot(gs[3, 1]), no_mod,
                       dist_matrices, thresh_list, CONCEPTS, fits,
                       "F₂.  Steepness vs Fit (excl. Modularity)",
                       show_regression=True)

    # Panel G
    _bar_multimetric(fig.add_subplot(gs[3, 0]), dist_matrices, CONCEPTS)

    # Panels H, I
    _dotplot_bestfit(fig.add_subplot(gs[4, 0]), dist_matrices, thresh_list,
                     CONCEPTS, retention, thresh_desc, mode="threshold",
                     title="H.  Best-Fit Threshold per Concept")
    _dotplot_bestfit(fig.add_subplot(gs[4, 1]), dist_matrices, thresh_list,
                     CONCEPTS, retention, thresh_desc, mode="retention",
                     title="I.  Best-Fit Edge Retention per Concept")

    # Global legend + method note
    handles = [plt.Line2D([0], [0], color=cc[c], lw=2.5, label=c)
               for c in CONCEPTS]
    fig.legend(handles=handles, loc="upper center", ncol=min(n, 8),
               fontsize=9, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 0.975))

    fig.text(0.5, 0.955,
             f"Method: {method_name} — LLM values averaged across all available "
             f"dimensionalities ({', '.join(str(d) for d in DIMS)})\n"
             "Distance: |z_LLM − z_Human|  where  "
             "z = (value − mean) / std  (independently scaled across concepts "
             "at each threshold)\n"
             "Percolation computed from 256-d thresholded matrices.",
             ha="center", va="top", fontsize=8.5, color="#444444",
             fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7",
                       ec="#cccccc", alpha=0.9))

    fig.suptitle(
        f"Human Network Alignment — {method_name}",
        fontsize=20, fontweight="bold", y=0.998)

    savefig(fig, f"human_alignment_{method_name}.png",
            dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")


def plot_all_methods_human_alignment():
    """Entry point: generate one figure per correlation method."""
    apply_style()
    for method_name, cfg in METHODS.items():
        print(f"\n  Human alignment — {method_name}")
        try:
            _plot_human_alignment(method_name, cfg)
        except Exception as exc:
            import traceback
            print(f"    ERROR for {method_name}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    plot_all_methods_human_alignment()
