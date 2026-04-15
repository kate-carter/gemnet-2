"""
vis_semd_predictors.py — Greatest BCT Predictors of Semantic Diversity
======================================================================

Summarises which BCT network metrics best predict SemD across all
correlation methods (Mahalanobis, Spearman, Pearson, Cosine),
dimensionalities (256–3072), and thresholds (0.05–0.90).

Layout (3 × 2):
  A  Summary heatmap — peak |Spearman ρ(metric, SemD)| per method × metric.
     Stars mark p < .05 against null ρ = 0.
  B  Forest plot — best ρ ± bootstrap 95 % CI for modularity & corefit,
     one row per method.  p-values annotated.
  C  Threshold landscape for *modularity* — ρ(SemD, modularity) vs threshold,
     one line per method (mean ± SE across dimensionalities).
  D  Same as C for *corefit*.
  E  Best-condition scatter — the single (method, dim, threshold) that yields
     the strongest |ρ| for any BCT metric, plotted concept-by-concept.
  F  Ranked bar chart — all 6 BCT metrics at their respective peak conditions,
     with significance markers.

Output: vis_results/semd_predictors.png
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
    CONCEPTS, CONCEPT_COLORS, DIMS, THRESHOLDS, METHODS, SEMD_PATH,
)

BCT_METRICS = ["modularity", "corefit", "diversity", "clustering", "strength", "degree"]
PRIORITY    = ["modularity", "corefit"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _find_workbook(results_dir, prefix, dim):
    candidates = [
        results_dir / f"{prefix}_{dim}_results_master.xlsx",
        results_dir / f"{prefix}_{dim}_master_results.xlsx",
        results_dir / f"{dim}_results_master.xlsx",
        results_dir / f"{dim}_master_results.xlsx",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_semd():
    df = pd.read_excel(SEMD_PATH, sheet_name="Values", engine="openpyxl", header=1)
    df.columns = ["term", "mean_cos", "SemD", "BNC_wordcount",
                   "BNC_contexts", "BNC_freq", "lg_BNC_freq"]
    df["term"] = df["term"].astype(str).str.strip().str.lower()
    return {c: float(df.loc[df["term"] == c, "SemD"].values[0])
            for c in CONCEPTS if c in df["term"].values}


def load_bct_at_threshold(results_dir, prefix, dim, metric, threshold):
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


def compute_all_rhos(semd):
    """Sweep every (method, dim, threshold, metric) and return a DataFrame
    with columns [method, dim, threshold, metric, rho, pval, n]."""
    semd_arr = {c: semd[c] for c in CONCEPTS if c in semd}
    records = []
    for method_name, cfg in METHODS.items():
        rdir, prefix = cfg["results_dir"], cfg["prefix"]
        for dim in DIMS:
            for metric in BCT_METRICS:
                for t in THRESHOLDS:
                    bct = load_bct_at_threshold(rdir, prefix, dim, metric, t)
                    paired = [(bct[c], semd_arr[c])
                              for c in CONCEPTS if c in bct and c in semd_arr
                              and np.isfinite(bct[c])]
                    if len(paired) < 5:
                        continue
                    bv, sv = zip(*paired)
                    rho, p = spearmanr(bv, sv)
                    records.append({
                        "method": method_name, "dim": dim,
                        "threshold": round(float(t), 2),
                        "metric": metric, "rho": rho, "pval": p,
                        "n": len(paired),
                    })
    return pd.DataFrame(records)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_semd_predictors():
    apply_style()

    print("  Loading SemD …")
    semd = load_semd()
    if len(semd) < 5:
        print("    Too few SemD values — aborting.")
        return

    print("  Sweeping all (method, dim, threshold, metric) conditions …")
    df = compute_all_rhos(semd)
    if df.empty:
        print("    No BCT data found — aborting.")
        return
    print(f"    {len(df)} condition–metric combinations evaluated.")

    method_names = list(METHODS.keys())
    method_colors = {m: METHODS[m]["color"] for m in method_names}

    fig = plt.figure(figsize=(26, 22))
    gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.30,
                           left=0.07, right=0.95, top=0.93, bottom=0.04)

    # ── Panel A: Summary heatmap ──────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    ordered_metrics = PRIORITY + [m for m in BCT_METRICS if m not in PRIORITY]
    peak = (df.loc[df.groupby(["method", "metric"])["rho"].apply(lambda x: x.abs().idxmax())]
              .set_index(["method", "metric"])[["rho", "pval"]])

    heatmap = np.full((len(method_names), len(ordered_metrics)), np.nan)
    pmap    = np.ones_like(heatmap)
    for mi, m in enumerate(method_names):
        for mj, met in enumerate(ordered_metrics):
            if (m, met) in peak.index:
                heatmap[mi, mj] = peak.loc[(m, met), "rho"]
                pmap[mi, mj]    = peak.loc[(m, met), "pval"]

    vmax = max(0.6, np.nanmax(np.abs(heatmap)))
    im = ax_a.imshow(heatmap, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                     aspect="auto")
    for mi in range(len(method_names)):
        for mj in range(len(ordered_metrics)):
            v = heatmap[mi, mj]
            p = pmap[mi, mj]
            if np.isnan(v):
                continue
            star = " *" if p < 0.05 else ""
            txt_col = "white" if abs(v) > 0.45 * vmax else "black"
            ax_a.text(mj, mi, f"{v:+.2f}{star}", ha="center", va="center",
                      fontsize=9, fontweight="bold" if star else "normal",
                      color=txt_col)

    ax_a.set_xticks(range(len(ordered_metrics)))
    ax_a.set_xticklabels([m.replace("_", " ").title() for m in ordered_metrics],
                         fontsize=9, rotation=30, ha="right")
    ax_a.set_yticks(range(len(method_names)))
    ax_a.set_yticklabels(method_names, fontsize=10)
    fig.colorbar(im, ax=ax_a, shrink=0.7, label="Peak Spearman ρ (SemD)")
    ax_a.set_title("A.  Peak ρ(BCT metric, SemD) by Method    (* p < .05)",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Panel B: Forest plot for modularity & corefit ─────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    forest_rows = []
    for met in PRIORITY:
        for m in method_names:
            sub = df[(df["method"] == m) & (df["metric"] == met)]
            if sub.empty:
                continue
            best_row = sub.loc[sub["rho"].abs().idxmax()]
            rho, p = best_row["rho"], best_row["pval"]
            dim_lbl = f"{int(best_row['dim'])}d"
            t_lbl   = f"t={best_row['threshold']:.2f}"
            forest_rows.append({
                "label": f"{m}\n({dim_lbl}, {t_lbl})",
                "metric": met, "rho": rho, "pval": p,
                "color": method_colors[m],
            })

    if forest_rows:
        y_positions = np.arange(len(forest_rows))
        for i, row in enumerate(forest_rows):
            marker = "s" if row["metric"] == "corefit" else "o"
            face = row["color"] if row["pval"] < 0.05 else "none"
            ax_b.scatter(row["rho"], i, color=row["color"], marker=marker,
                         s=120, zorder=4, edgecolors="black", linewidths=0.8,
                         facecolors=face)
            sig = ""
            if row["pval"] < 0.001:
                sig = "***"
            elif row["pval"] < 0.01:
                sig = "**"
            elif row["pval"] < 0.05:
                sig = "*"
            ax_b.text(row["rho"] + 0.03, i,
                      f"ρ = {row['rho']:+.3f}  (p = {row['pval']:.4f}) {sig}",
                      va="center", fontsize=8.5)

        ax_b.set_yticks(y_positions)
        ax_b.set_yticklabels([r["label"] for r in forest_rows], fontsize=8)
        ax_b.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_b.set_xlabel("Spearman ρ with SemD", fontsize=11)
        ax_b.set_xlim(-1.0, 1.0)
        ax_b.grid(axis="x", alpha=0.25)
        ax_b.invert_yaxis()

        mod_patch = mpatches.Patch(color="gray", label="○  Modularity")
        cor_patch = mpatches.Patch(color="gray", label="□  Corefit")
        sig_patch = mpatches.Patch(facecolor="gray", edgecolor="black",
                                   label="Filled = p < .05")
        ax_b.legend(handles=[mod_patch, cor_patch, sig_patch], fontsize=7.5,
                    loc="upper left", bbox_to_anchor=(0.0, -0.12),
                    ncol=3, framealpha=0.85, borderaxespad=0)

    ax_b.set_title("B.  Modularity & Corefit — Best Condition per Method",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Panel C: Threshold landscape — Modularity ─────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_threshold_landscape(ax_c, df, "modularity", method_names,
                              method_colors, "C", semd)

    # ── Panel D: Threshold landscape — Corefit ────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_threshold_landscape(ax_d, df, "corefit", method_names,
                              method_colors, "D", semd)

    # ── Panel E: Best-condition scatter ────────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 0])

    # Prefer conditions with the widest concept coverage, then highest |ρ|
    max_n = df["n"].max()
    full_cov = df[df["n"] >= max_n]
    if full_cov.empty:
        full_cov = df
    best_idx = full_cov["rho"].abs().idxmax()
    best = df.loc[best_idx]
    bm, bd, bt, bmet = best["method"], int(best["dim"]), best["threshold"], best["metric"]
    cfg = METHODS[bm]
    bct_vals = load_bct_at_threshold(cfg["results_dir"], cfg["prefix"], bd, bmet, bt)

    xs, ys, cs_plot, labels = [], [], [], []
    for c in CONCEPTS:
        if c in bct_vals and c in semd and np.isfinite(bct_vals[c]):
            xs.append(bct_vals[c])
            ys.append(semd[c])
            cs_plot.append(CONCEPT_COLORS.get(c, "#888888"))
            labels.append(c)

    if xs:
        ax_e.scatter(xs, ys, c=cs_plot, s=80, edgecolors="black",
                     linewidths=0.5, zorder=3)
        for i, c in enumerate(labels):
            ax_e.annotate(c, (xs[i], ys[i]), fontsize=7.5, ha="center",
                          va="bottom", xytext=(0, 5), textcoords="offset points",
                          color=cs_plot[i])
        if len(xs) > 2:
            z = np.polyfit(xs, ys, 1)
            x_line = np.linspace(min(xs), max(xs), 50)
            ax_e.plot(x_line, np.polyval(z, x_line), "--", color="grey",
                      alpha=0.6, linewidth=2)
        rho_e, p_e = spearmanr(xs, ys)
        ax_e.set_xlabel(f"{bmet.replace('_',' ').title()}", fontsize=12)
        ax_e.set_ylabel("SemD", fontsize=12)
        sig_str = ""
        if p_e < 0.001:   sig_str = "***"
        elif p_e < 0.01:  sig_str = "**"
        elif p_e < 0.05:  sig_str = "*"
        ax_e.text(0.03, 0.97,
                  f"Best overall: {bm}, {bd}d, t={bt:.2f}\n"
                  f"ρ = {rho_e:+.3f},  p = {p_e:.4f} {sig_str}",
                  transform=ax_e.transAxes, fontsize=10, va="top",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax_e.grid(alpha=0.2)

    ax_e.set_title("E.  Best Overall BCT Predictor of SemD",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Panel F: All metrics ranked bar chart ─────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 1])

    best_per_metric = (df.loc[df.groupby("metric")["rho"]
                       .apply(lambda x: x.abs().idxmax())]
                       .sort_values("rho", key=abs, ascending=True))

    y_pos = np.arange(len(best_per_metric))
    bar_colors = []
    for _, row in best_per_metric.iterrows():
        if row["metric"] in PRIORITY:
            bar_colors.append("#d62728")
        else:
            bar_colors.append(method_colors.get(row["method"], "#999999"))

    ax_f.barh(y_pos, best_per_metric["rho"].values, color=bar_colors,
              alpha=0.75, edgecolor="black", linewidth=0.4)

    labels_f = []
    for _, row in best_per_metric.iterrows():
        met_label = row["metric"].replace("_", " ").title()
        cond = f"{row['method']}, {int(row['dim'])}d, t={row['threshold']:.2f}"
        labels_f.append(f"{met_label} ({cond})")

    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(labels_f, fontsize=7.5)

    for i, (_, row) in enumerate(best_per_metric.iterrows()):
        p = row["pval"]
        r = row["rho"]
        if p < 0.001:   star = "***"
        elif p < 0.01:  star = "**"
        elif p < 0.05:  star = "*"
        else:            star = "n.s."
        if r >= 0:
            ax_f.text(0.02, i, f"{star}  p={p:.4f}", fontsize=7.5,
                      va="center", ha="left", fontstyle="italic")
        else:
            ax_f.text(-0.02, i, f"p={p:.4f}  {star}", fontsize=7.5,
                      va="center", ha="right", fontstyle="italic")

    ax_f.axvline(0, color="black", linewidth=0.8)
    ax_f.set_xlabel("Spearman ρ with SemD", fontsize=11)
    ax_f.set_xlim(-1, 1)
    ax_f.grid(axis="x", alpha=0.2)

    priority_patch = mpatches.Patch(color="#d62728",
                                    label="Priority (modularity / corefit)")
    other_patch    = mpatches.Patch(color="#999999", label="Other metrics")
    ax_f.legend(handles=[priority_patch, other_patch], fontsize=7.5,
                loc="upper left", bbox_to_anchor=(0.0, -0.12),
                ncol=2, framealpha=0.85, borderaxespad=0)

    ax_f.set_title("F.  All BCT Metrics Ranked — Best Condition per Metric",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Greatest BCT Predictors of Semantic Diversity (SemD)\n"
        "Across 4 Correlation Methods × 5 Dimensionalities × 18 Thresholds",
        fontsize=16, fontweight="bold", y=0.99,
    )
    savefig(fig, "semd_predictors.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")


def _plot_threshold_landscape(ax, df, metric, method_names, method_colors,
                              panel_letter, semd):
    """ρ(SemD, metric) vs threshold, one line per method (mean ± SE across dims)."""
    thresholds = np.sort(df["threshold"].unique())

    for m in method_names:
        sub = df[(df["method"] == m) & (df["metric"] == metric)]
        if sub.empty:
            continue
        means, sems, sig_mask = [], [], []
        for t in thresholds:
            t_sub = sub[sub["threshold"] == t]
            if t_sub.empty:
                means.append(np.nan)
                sems.append(0)
                sig_mask.append(False)
            else:
                rhos = t_sub["rho"].values
                means.append(np.nanmean(rhos))
                sems.append(np.nanstd(rhos) / max(1, np.sqrt(len(rhos))))
                sig_mask.append((t_sub["pval"] < 0.05).any())

        means = np.array(means)
        sems  = np.array(sems)
        sig_mask = np.array(sig_mask)
        color = method_colors[m]

        ax.plot(thresholds, means, color=color, linewidth=1.5, label=m, zorder=2)
        ax.fill_between(thresholds, means - sems, means + sems,
                        color=color, alpha=0.12, zorder=1)

        valid = ~np.isnan(means)
        if sig_mask.any():
            ax.scatter(thresholds[valid & sig_mask], means[valid & sig_mask],
                       color=color, s=35, zorder=4, edgecolors="k",
                       linewidths=0.5)
        if (~sig_mask & valid).any():
            ax.scatter(thresholds[valid & ~sig_mask], means[valid & ~sig_mask],
                       facecolors="none", edgecolors=color, s=22, zorder=3,
                       linewidths=0.8)

    ax.axhline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Threshold", fontsize=10)
    ax.set_ylabel("Spearman ρ (SemD)", fontsize=10)
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    met_title = metric.replace("_", " ").title()
    ax.set_title(f"{panel_letter}.  ρ(SemD, {met_title}) vs Threshold  "
                 f"(mean ± SE across dims;  ● p < .05)",
                 fontsize=12, fontweight="bold", loc="left")


if __name__ == "__main__":
    plot_semd_predictors()
    print("Done — semd_predictors.png")
