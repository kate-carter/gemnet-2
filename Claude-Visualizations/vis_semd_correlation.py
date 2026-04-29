"""
vis_semd_correlation.py
───────────────────────
Spearman ρ between LLM-derived BCT statistics and Semantic Diversity (SemD),
computed across the 15 concepts at every threshold.

Layout
------
  Rows : BCT metric  (modularity | corefit | diversity | clustering | strength | degree)
  Cols : dimensionality (256 | 512 | 768 | 1536 | 3072)

Each subplot shows one line per correlation method (Mahalanobis, Spearman,
Pearson, Cosine).  Filled markers = p < .05 (significant vs. null ρ = 0);
open markers = non-significant.  Legend includes per-method mean ± SD of ρ
across thresholds.

Output saved to:
  /Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/
      gemnet-2/gemini-2/vis_results/semd_correlation.png
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from scipy import stats

from vis_config import (
    BASE, CONCEPTS, DIMS, THRESHOLDS, METHODS, SEMD_PATH, apply_style,
)

# ── Output ──────────────────────────────────────────────────────────────────
VIS_RESULTS_DIR = BASE / "gemnet-2/gemini-2/vis_results"
VIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["modularity", "corefit", "diversity", "clustering", "strength", "degree"]


# ── Data loading ─────────────────────────────────────────────────────────────

def _find_workbook(results_dir: Path, prefix: str, dim: int) -> Path | None:
    """Locate the master workbook for a given method/dim, trying common patterns."""
    candidates = [
        results_dir / f"{prefix}_{dim}_results_master.xlsx",
        results_dir / f"{prefix}_{dim}_master_results.xlsx",
        results_dir / f"{dim}_results_master.xlsx",
        results_dir / f"{dim}_master_results.xlsx",
    ]
    return next((p for p in candidates if p.exists()), None)


def _pick_sheet(sheets: dict[str, pd.DataFrame], name: str) -> pd.DataFrame | None:
    """Return sheet by name, trying '{name}18' variant as fallback."""
    df = sheets.get(name)
    if df is None:
        df = sheets.get(f"{name}18")
    return df


def load_semd() -> np.ndarray | None:
    """
    Load per-concept SemD values from SemD_data.xlsx.
    Returns array(n_concepts) aligned to CONCEPTS ordering, or None on failure.
    """
    if not SEMD_PATH.exists():
        print(f"  ERROR: {SEMD_PATH} not found.")
        return None
    try:
        df = pd.read_excel(SEMD_PATH, sheet_name="Values",
                           engine="openpyxl", header=1)
        df.columns = ["term", "mean_cos", "SemD", "BNC_wordcount",
                       "BNC_contexts", "BNC_freq", "lg_BNC_freq"]
        df["term"] = df["term"].astype(str).str.strip().str.lower()

        vals = np.full(len(CONCEPTS), np.nan)
        for i, c in enumerate(CONCEPTS):
            match = df.loc[df["term"] == c, "SemD"]
            if len(match):
                vals[i] = float(match.values[0])

        n_valid = np.isfinite(vals).sum()
        print(f"  Loaded SemD for {n_valid}/{len(CONCEPTS)} concepts")
        if n_valid < 5:
            return None
        return vals
    except Exception as exc:
        print(f"  ERROR loading SemD_data.xlsx: {exc}")
        return None


def compute_rho_matrix(
    method_name: str,
    dim: int,
    metric: str,
    semd_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For one (method, dim, metric) combination, return:
        rhos  : Spearman ρ at each threshold  shape (n_thresholds,)
        pvals : two-tailed p-value            shape (n_thresholds,)
    NaN where data is missing.
    """
    cfg = METHODS[method_name]
    wb_path = _find_workbook(cfg["results_dir"], cfg["prefix"], dim)

    n = len(THRESHOLDS)
    rhos  = np.full(n, np.nan)
    pvals = np.full(n, np.nan)

    if wb_path is None:
        return rhos, pvals

    try:
        xls    = pd.ExcelFile(wb_path, engine="openpyxl")
        sheets = {s: xls.parse(s) for s in xls.sheet_names}
        df     = _pick_sheet(sheets, metric)
        if df is None:
            return rhos, pvals

        if "threshold" not in df.columns:
            first_col = df.columns[0]
            if first_col != "threshold":
                df = df.rename(columns={first_col: "threshold"})
            else:
                return rhos, pvals
        thresh_col = df["threshold"].round(2).values
        concept_cols = [c for c in CONCEPTS if c in df.columns]

        for ti, t in enumerate(THRESHOLDS):
            t_rounded = round(float(t), 2)
            row_mask = thresh_col == t_rounded
            if not row_mask.any():
                continue
            row = df.loc[row_mask].iloc[0]
            llm_vals = np.array([row[c] for c in concept_cols], dtype=float)
            if len(concept_cols) < len(CONCEPTS):
                full = np.full(len(CONCEPTS), np.nan)
                for ci_idx, c in enumerate(CONCEPTS):
                    if c in concept_cols:
                        full[ci_idx] = llm_vals[concept_cols.index(c)]
                llm_vals = full

            mask = np.isfinite(llm_vals) & np.isfinite(semd_vals)
            if mask.sum() < 3:
                continue

            rho, p = stats.spearmanr(llm_vals[mask], semd_vals[mask])
            rhos[ti]  = rho
            pvals[ti] = p

    except Exception as exc:
        print(f"  WARNING: {method_name} {dim}-d {metric} — {exc}")

    return rhos, pvals


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_semd_correlation() -> None:
    apply_style()

    print("Loading SemD values…")
    semd_vals = load_semd()
    if semd_vals is None:
        print("  No SemD data — aborting.")
        return

    n_metrics = len(METRICS)
    n_dims    = len(DIMS)

    fig, axes = plt.subplots(
        n_metrics, n_dims,
        figsize=(4.0 * n_dims, 3.2 * n_metrics),
        sharey="row",
        constrained_layout=True,
    )
    fig.suptitle(
        "Spearman ρ (BCT metric vs. SemD) across 15 concepts",
        fontsize=14, fontweight="bold",
    )

    x = THRESHOLDS

    for ri, metric in enumerate(METRICS):
        for ci, dim in enumerate(DIMS):
            ax = axes[ri, ci]

            for method_name, cfg in METHODS.items():
                rhos, pvals = compute_rho_matrix(method_name, dim, metric, semd_vals)

                if np.all(np.isnan(rhos)):
                    continue

                color  = cfg["color"]
                marker = cfg["marker"]

                ax.plot(x, rhos, color=color, linewidth=1.3, zorder=2, alpha=0.7)

                sig_mask  = (~np.isnan(pvals)) & (pvals < 0.05)
                nsig_mask = (~np.isnan(pvals)) & (pvals >= 0.05)

                if sig_mask.any():
                    ax.scatter(
                        x[sig_mask], rhos[sig_mask],
                        color=color, marker=marker,
                        s=40, zorder=4, linewidths=0.5, edgecolors="k",
                    )
                if nsig_mask.any():
                    ax.scatter(
                        x[nsig_mask], rhos[nsig_mask],
                        facecolors="none", edgecolors=color, marker=marker,
                        s=28, zorder=3, linewidths=1.0,
                    )

                valid = rhos[~np.isnan(rhos)]
                if len(valid):
                    lbl = f"{method_name}  (μ={np.mean(valid):+.2f}, σ={np.std(valid):.2f})"
                else:
                    lbl = method_name
                ax.plot([], [], color=color, marker=marker,
                        linewidth=1.3, markersize=5, label=lbl)

            ax.axhline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.4)
            ax.set_xlim(x[0] - 0.03, x[-1] + 0.03)

            ax.set_xticks(x[::2])
            ax.set_xticklabels([f"{v:.2f}" for v in x[::2]], rotation=45, fontsize=7)

            if ci == 0:
                ax.set_ylabel(f"{metric.capitalize()}\nSpearman ρ", fontsize=10)
            if ri == 0:
                ax.set_title(f"{dim}-d", fontsize=11, fontweight="bold")
            if ri == n_metrics - 1:
                ax.set_xlabel("Threshold", fontsize=9)

            ax.grid(axis="y", linewidth=0.4, alpha=0.4)
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

    sig_patch  = mlines.Line2D([], [], color="k", marker="o", markersize=6,
                               linewidth=0, label="p < .05 (filled)")
    nsig_patch = mlines.Line2D([], [], color="k", marker="o", markersize=6,
                               linewidth=0, markerfacecolor="none", label="p ≥ .05 (open)")
    fig.legend(
        handles=[sig_patch, nsig_patch],
        loc="lower center", ncol=2, fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
        title="Significance (vs. ρ = 0)",
        title_fontsize=8,
    )

    out_path = VIS_RESULTS_DIR / "semd_correlation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    plot_semd_correlation()
