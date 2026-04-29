"""
vis_dual_predictors.py — Joint Predictors of Human BCT & Semantic Diversity
============================================================================

Identifies which BCT metrics, correlation methods, and pipeline conditions
best predict *both* human network statistics and SemD simultaneously.

Sweeps all methods × dimensionalities × thresholds × BCT metrics and
computes Spearman ρ against both targets in one pass.

Layout (3 × 2):
  A  Human heatmap — peak |ρ(LLM, Human)| per method × metric.
  B  SemD heatmap — peak |ρ(BCT, SemD)| per method × metric.
     A and B share the same colour scale for direct comparison.
  C  Combined strength heatmap — min(|ρ_human|, |ρ_semd|) per method × metric.
     Stars where *both* p-values < .05.
  D  Best dual-predictor bar chart — for each metric, the condition that
     maximises min(|ρ_human|, |ρ_semd|), with significance markers.
  E  Method ranking — average dual strength across metrics.
  F  Metric ranking — average dual strength across methods.

Output: vis_results/dual_predictors.png
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

from vis_config import (
    apply_style, savefig, OUTPUT_DIR,
    CONCEPTS, DIMS, THRESHOLDS, METHODS,
    HUMAN_STATS_PATH, SEMD_PATH,
)

BCT_METRICS = ["modularity", "corefit", "diversity", "clustering", "strength", "degree"]

HUMAN_ROW_MAP = {
    "modularity": "hum_modularity",
    "corefit":    "hum_corefit",
    "diversity":  "hum_diversity",
    "clustering": "hum_clustering",
    "strength":   "hum_strength",
    "degree":     "hum_degree",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _find_workbook(results_dir, prefix, dim):
    candidates = [
        results_dir / f"{prefix}_{dim}_results_master.xlsx",
        results_dir / f"{prefix}_{dim}_master_results.xlsx",
        results_dir / f"{dim}_results_master.xlsx",
        results_dir / f"{dim}_master_results.xlsx",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_human_baselines():
    """Return {metric: {concept: value}} from human_stats.xlsx."""
    if not HUMAN_STATS_PATH.exists():
        print(f"  ERROR: {HUMAN_STATS_PATH} not found.")
        return {}
    df = pd.read_excel(HUMAN_STATS_PATH, sheet_name="human_bct", engine="openpyxl")
    df = df.rename(columns={df.columns[0]: "concept"}).set_index("concept")
    baselines = {}
    for metric, row_key in HUMAN_ROW_MAP.items():
        if row_key in df.index:
            vals = {}
            for c in CONCEPTS:
                if c in df.columns:
                    v = df.loc[row_key, c]
                    if pd.notna(v):
                        vals[c] = float(v)
            if vals:
                baselines[metric] = vals
    print(f"  Loaded human baselines: {list(baselines.keys())}")
    return baselines


def load_semd():
    """Return {concept: SemD_value}."""
    df = pd.read_excel(SEMD_PATH, sheet_name="Values", engine="openpyxl", header=1)
    df.columns = ["term", "mean_cos", "SemD", "BNC_wordcount",
                   "BNC_contexts", "BNC_freq", "lg_BNC_freq"]
    df["term"] = df["term"].astype(str).str.strip().str.lower()
    semd = {c: float(df.loc[df["term"] == c, "SemD"].values[0])
            for c in CONCEPTS if c in df["term"].values}
    print(f"  Loaded SemD for {len(semd)} concepts")
    return semd


def _load_llm_at_threshold(results_dir, prefix, dim, metric, threshold):
    """Return {concept: value} for one (dim, metric, threshold)."""
    path = _find_workbook(results_dir, prefix, dim)
    if path is None:
        return {}
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet_name = None
    for s in xls.sheet_names:
        if s.strip().lower() == metric or s.strip().lower() == f"{metric}18":
            sheet_name = s
            break
    if sheet_name is None:
        return {}
    df = xls.parse(sheet_name)
    if "threshold" not in df.columns:
        df = df.rename(columns={df.columns[0]: "threshold"})
    t_round = round(float(threshold), 2)
    row = df[df["threshold"].round(2) == t_round]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {c: float(row[c]) for c in CONCEPTS
            if c in df.columns and pd.notna(row[c])}


def compute_dual_rhos(human, semd):
    """Sweep every condition and compute ρ against both human and SemD."""
    records = []
    for method_name, cfg in METHODS.items():
        rdir, prefix = cfg["results_dir"], cfg["prefix"]
        for dim in DIMS:
            for metric in BCT_METRICS:
                hvals = human.get(metric, {})
                for t in THRESHOLDS:
                    llm = _load_llm_at_threshold(rdir, prefix, dim, metric, t)
                    if not llm:
                        continue

                    rec = {"method": method_name, "dim": dim,
                           "threshold": round(float(t), 2), "metric": metric}

                    # ρ vs human
                    h_paired = [(llm[c], hvals[c]) for c in CONCEPTS
                                if c in llm and c in hvals and np.isfinite(llm[c])]
                    if len(h_paired) >= 5:
                        lv, hv = zip(*h_paired)
                        rho_h, p_h = spearmanr(lv, hv)
                        rec.update(rho_human=rho_h, p_human=p_h, n_human=len(h_paired))
                    else:
                        rec.update(rho_human=np.nan, p_human=np.nan, n_human=0)

                    # ρ vs SemD
                    s_paired = [(llm[c], semd[c]) for c in CONCEPTS
                                if c in llm and c in semd and np.isfinite(llm[c])]
                    if len(s_paired) >= 5:
                        lv, sv = zip(*s_paired)
                        rho_s, p_s = spearmanr(lv, sv)
                        rec.update(rho_semd=rho_s, p_semd=p_s, n_semd=len(s_paired))
                    else:
                        rec.update(rho_semd=np.nan, p_semd=np.nan, n_semd=0)

                    records.append(rec)

    df = pd.DataFrame(records)
    df["dual_strength"] = np.minimum(df["rho_human"].abs(), df["rho_semd"].abs())
    return df


# ── Heatmap helper ────────────────────────────────────────────────────────────

def _draw_heatmap(ax, matrix, pmatrix, method_names, metric_names, vmax,
                  cmap, label, title, fig, both_sig=False, pmatrix2=None):
    """Draw an annotated heatmap on *ax*."""
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for mi in range(matrix.shape[0]):
        for mj in range(matrix.shape[1]):
            v = matrix[mi, mj]
            if np.isnan(v):
                continue
            if both_sig and pmatrix2 is not None:
                star = " *" if (pmatrix[mi, mj] < 0.05 and pmatrix2[mi, mj] < 0.05) else ""
            else:
                star = " *" if pmatrix[mi, mj] < 0.05 else ""
            txt_col = "white" if abs(v) > 0.45 * vmax else "black"
            ax.text(mj, mi, f"{v:+.2f}{star}", ha="center", va="center",
                    fontsize=9, fontweight="bold" if star else "normal",
                    color=txt_col)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names],
                       fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label=label)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_dual_predictors():
    apply_style()

    print("  Loading human baselines …")
    human = load_human_baselines()
    if not human:
        print("    No human data — aborting.")
        return

    print("  Loading SemD …")
    semd = load_semd()
    if len(semd) < 5:
        print("    Too few SemD values — aborting.")
        return

    print("  Sweeping all conditions (dual targets) …")
    df = compute_dual_rhos(human, semd)
    if df.empty:
        print("    No data — aborting.")
        return
    print(f"    {len(df)} condition–metric combinations evaluated.")

    method_names  = list(METHODS.keys())
    method_colors = {m: METHODS[m]["color"] for m in method_names}
    ordered_metrics = BCT_METRICS

    fig = plt.figure(figsize=(26, 24))
    gs = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.30,
                           left=0.07, right=0.95, top=0.92, bottom=0.04)

    # ── Compute peak-rho tables for heatmaps ─────────────────────────────────
    def _peak_table(rho_col, p_col):
        """Return (value_matrix, pval_matrix) of peak |ρ| per method × metric."""
        vals = np.full((len(method_names), len(ordered_metrics)), np.nan)
        ps   = np.ones_like(vals)
        valid = df.dropna(subset=[rho_col])
        if valid.empty:
            return vals, ps
        idx = valid.groupby(["method", "metric"])[rho_col].apply(
            lambda x: x.abs().idxmax())
        peak = valid.loc[idx].set_index(["method", "metric"])[[rho_col, p_col]]
        for mi, m in enumerate(method_names):
            for mj, met in enumerate(ordered_metrics):
                if (m, met) in peak.index:
                    vals[mi, mj] = peak.loc[(m, met), rho_col]
                    ps[mi, mj]   = peak.loc[(m, met), p_col]
        return vals, ps

    h_vals, h_ps = _peak_table("rho_human", "p_human")
    s_vals, s_ps = _peak_table("rho_semd",  "p_semd")

    # Combined: min(|ρ_h|, |ρ_s|) with sign of ρ_h (arbitrary convention)
    dual_vals = np.where(
        np.abs(h_vals) < np.abs(s_vals), h_vals, s_vals)
    # For dual, NaN if either is NaN
    dual_vals = np.where(np.isnan(h_vals) | np.isnan(s_vals), np.nan, dual_vals)

    vmax = max(0.6, np.nanmax(np.abs(h_vals)),
               np.nanmax(np.abs(s_vals)))

    # ── Panel A: Human heatmap ───────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_heatmap(ax_a, h_vals, h_ps, method_names, ordered_metrics, vmax,
                  "RdBu_r", "Peak Spearman ρ",
                  "A.  Peak ρ(LLM, Human)    (* p < .05)", fig)

    # ── Panel B: SemD heatmap ────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_heatmap(ax_b, s_vals, s_ps, method_names, ordered_metrics, vmax,
                  "RdBu_r", "Peak Spearman ρ",
                  "B.  Peak ρ(BCT, SemD)    (* p < .05)", fig)

    # ── Panel C: Combined strength heatmap ───────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    _draw_heatmap(ax_c, dual_vals, h_ps, method_names, ordered_metrics, vmax,
                  "RdBu_r", "min(|ρ_human|, |ρ_semd|) with sign",
                  "C.  Dual Strength    (* both p < .05)",
                  fig, both_sig=True, pmatrix2=s_ps)

    # ── Panel D: Best dual-predictor bar chart ───────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    valid_dual = df.dropna(subset=["rho_human", "rho_semd"])
    if not valid_dual.empty:
        best_dual = valid_dual.loc[
            valid_dual.groupby("metric")["dual_strength"].idxmax()
        ].sort_values("dual_strength", ascending=True)

        y_pos = np.arange(len(best_dual))
        bar_colors = [method_colors.get(row["method"], "#999999")
                      for _, row in best_dual.iterrows()]
        ax_d.barh(y_pos, best_dual["dual_strength"].values,
                  color=bar_colors, alpha=0.75, edgecolor="black", linewidth=0.4)

        labels_d = []
        for _, row in best_dual.iterrows():
            met_label = row["metric"].replace("_", " ").title()
            cond = f"{row['method']}, {int(row['dim'])}d, t={row['threshold']:.2f}"
            labels_d.append(f"{met_label}\n({cond})")

        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels(labels_d, fontsize=8)

        for i, (_, row) in enumerate(best_dual.iterrows()):
            ph, ps = row["p_human"], row["p_semd"]
            def _star(p):
                if p < 0.001: return "***"
                if p < 0.01:  return "**"
                if p < 0.05:  return "*"
                return "n.s."
            label_txt = f"H:{_star(ph)}  S:{_star(ps)}"
            ax_d.text(row["dual_strength"] + 0.01, i, label_txt,
                      va="center", fontsize=7.5, fontstyle="italic")

        ax_d.set_xlabel("Dual Strength  min(|ρ_human|, |ρ_semd|)", fontsize=10)
        ax_d.set_xlim(0, 1)
        ax_d.grid(axis="x", alpha=0.25)

    ax_d.set_title("D.  Best Dual-Predictor Condition per Metric",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Panel E: Method ranking ──────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 0])

    if not valid_dual.empty:
        # For each (method, metric) pick the condition with best dual_strength
        best_mm = valid_dual.loc[
            valid_dual.groupby(["method", "metric"])["dual_strength"].idxmax()]
        method_avg = (best_mm.groupby("method")["dual_strength"]
                      .agg(["mean", "std"]).reindex(method_names))
        method_avg = method_avg.sort_values("mean", ascending=True)

        y_pos_e = np.arange(len(method_avg))
        colors_e = [method_colors.get(m, "#999999") for m in method_avg.index]
        ax_e.barh(y_pos_e, method_avg["mean"].values, xerr=method_avg["std"].values,
                  color=colors_e, alpha=0.75, edgecolor="black", linewidth=0.4,
                  capsize=3)
        ax_e.set_yticks(y_pos_e)
        ax_e.set_yticklabels(method_avg.index, fontsize=10)
        ax_e.set_xlabel("Mean Dual Strength ± SD", fontsize=10)
        ax_e.set_xlim(0, 1)
        ax_e.grid(axis="x", alpha=0.25)

        for i, (m, row) in enumerate(method_avg.iterrows()):
            ax_e.text(row["mean"] + row["std"] + 0.02, i,
                      f"{row['mean']:.3f}", va="center", fontsize=9)

    ax_e.set_title("E.  Method Ranking — Mean Dual Strength across Metrics",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Panel F: Metric ranking ──────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 1])

    if not valid_dual.empty:
        best_mm2 = valid_dual.loc[
            valid_dual.groupby(["method", "metric"])["dual_strength"].idxmax()]
        metric_avg = (best_mm2.groupby("metric")["dual_strength"]
                      .agg(["mean", "std"]).reindex(BCT_METRICS))
        metric_avg = metric_avg.sort_values("mean", ascending=True)

        y_pos_f = np.arange(len(metric_avg))
        colors_f = ["#d62728" if m in ("modularity", "corefit") else "#4a90d9"
                    for m in metric_avg.index]
        ax_f.barh(y_pos_f, metric_avg["mean"].values, xerr=metric_avg["std"].values,
                  color=colors_f, alpha=0.75, edgecolor="black", linewidth=0.4,
                  capsize=3)
        ax_f.set_yticks(y_pos_f)
        ax_f.set_yticklabels([m.replace("_", " ").title() for m in metric_avg.index],
                             fontsize=10)
        ax_f.set_xlabel("Mean Dual Strength ± SD", fontsize=10)
        ax_f.set_xlim(0, 1)
        ax_f.grid(axis="x", alpha=0.25)

        for i, (m, row) in enumerate(metric_avg.iterrows()):
            ax_f.text(row["mean"] + row["std"] + 0.02, i,
                      f"{row['mean']:.3f}", va="center", fontsize=9)

        priority_patch = mpatches.Patch(color="#d62728",
                                        label="Priority (modularity / corefit)")
        other_patch    = mpatches.Patch(color="#4a90d9", label="Other metrics")
        ax_f.legend(handles=[priority_patch, other_patch], fontsize=8,
                    loc="lower right", framealpha=0.85)

    ax_f.set_title("F.  Metric Ranking — Mean Dual Strength across Methods",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Joint Predictors of Human BCT Values & Semantic Diversity\n"
        f"Across {len(METHODS)} Correlation Methods × "
        f"{len(DIMS)} Dimensionalities × {len(THRESHOLDS)} Thresholds",
        fontsize=16, fontweight="bold", y=0.99,
    )
    savefig(fig, "dual_predictors.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")


if __name__ == "__main__":
    plot_dual_predictors()
    print("Done — dual_predictors.png")
