#!/usr/bin/env python3
"""
Figure 4 – SemD (Semantic Diversity) integration with network properties.

Panels:
  A  SemD values for the 15 concepts (ranked bar chart)
  B  SemD vs network metric at best-match threshold (scatter + regression)
  C  Correlation landscape: Spearman r (SemD × metric) across thresholds
  D  SemD vs mean absolute correlation (embedding geometry, 256-d)
  E  SemD vs human network density
  F  SemD × BNC frequency, colored by concept
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from vis_config import (
    apply_style, savefig, CONCEPTS, CONCEPT_COLORS,
    SEMD_PATH, RESULTS_DIR, CORR_DIR, HUMAN_CSV_DIR, OUTPUT_DIR,
    load_human_network, METHODS,
)

apply_style()


def load_semd():
    """Load SemD for the 15 concepts."""
    df = pd.read_excel(SEMD_PATH, sheet_name="Values", engine="openpyxl", header=1)
    df.columns = ["term", "mean_cos", "SemD", "BNC_wordcount",
                   "BNC_contexts", "BNC_freq", "lg_BNC_freq"]
    semd = df[df["term"].isin(CONCEPTS)].copy()
    semd = semd.set_index("term")
    for col in semd.columns:
        semd[col] = pd.to_numeric(semd[col], errors="coerce")
    return semd


def plot_semd_network(method_name=None, results_dir=None,
                      corr_dir=None, corr_suffix=None):
    """Generate the SemD × network integration figure for one method.

    Called with no arguments uses the default Spearman paths (backward-compatible).
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if corr_dir is None:
        corr_dir = CORR_DIR
    if corr_suffix is None:
        corr_suffix = "spearman"
    label = method_name if method_name else "Spearman"

    semd = load_semd()

    # Locate 256-d results workbook
    candidates = (list(results_dir.glob("*_256_results_master.xlsx")) +
                  list(results_dir.glob("*_256_master_results.xlsx")) +
                  list(results_dir.glob("256_master_results.xlsx")))
    results_path = candidates[0] if candidates else None
    has_results  = results_path is not None and results_path.exists()
    sheets = {}
    if has_results:
        xls    = pd.ExcelFile(results_path, engine="openpyxl")
        sheets = {s: xls.parse(s) for s in xls.sheet_names}

    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    fig.suptitle(
        f"Semantic Diversity × Network Structure — {label}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ── A: SemD ranked bar chart ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ordered = semd.sort_values("SemD", ascending=True)
    y = range(len(ordered))
    colors = [CONCEPT_COLORS[c] for c in ordered.index if c in CONCEPT_COLORS]
    ax_a.barh(list(y)[:len(colors)], ordered["SemD"].values[:len(colors)],
              color=colors, edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(list(y)[:len(colors)])
    ax_a.set_yticklabels(list(ordered.index)[:len(colors)], fontsize=9)
    ax_a.set_xlabel("Semantic Diversity (SemD)")
    ax_a.set_title("A. SemD Values for 15 Concepts", fontweight="bold")

    # ── B: SemD vs modularity at a representative threshold ──────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if has_results:
        df_mod = sheets.get("modularity18") or sheets.get("modularity")
        if df_mod is not None:
            t_idx = 2  # ~threshold 0.15
            threshold_val = df_mod["threshold"].iloc[t_idx]
            mod_vals, semd_vals, concept_list = [], [], []
            for concept in CONCEPTS:
                if concept in df_mod.columns and concept in semd.index:
                    mod_vals.append(float(df_mod[concept].iloc[t_idx]))
                    semd_vals.append(float(semd.loc[concept, "SemD"]))
                    concept_list.append(concept)
            if mod_vals:
                ax_b.scatter(semd_vals, mod_vals,
                             c=[CONCEPT_COLORS[c] for c in concept_list],
                             s=60, edgecolors="black", linewidths=0.5, zorder=5)
                for i, c in enumerate(concept_list):
                    ax_b.annotate(c, (semd_vals[i], mod_vals[i]),
                                  fontsize=7, ha="left", va="bottom",
                                  xytext=(3, 3), textcoords="offset points")
                r, p = stats.spearmanr(semd_vals, mod_vals)
                ax_b.set_xlabel("SemD")
                ax_b.set_ylabel(f"Modularity (t={threshold_val:.2f})")
                ax_b.set_title(f"B. SemD vs Modularity (ρ={r:.3f}, p={p:.3f})",
                               fontweight="bold")
    if not ax_b.get_title():
        ax_b.set_title("B. SemD vs Modularity (no data)", fontweight="bold")

    # ── C: Spearman(SemD, metric) × threshold ────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    if has_results:
        metric_sheets = {
            "Diversity":  sheets.get("diversity18") or sheets.get("diversity"),
            "Modularity": sheets.get("modularity18") or sheets.get("modularity"),
            "Clustering": sheets.get("clustering"),
            "Core Fit":   sheets.get("corefit"),
            "Strength":   sheets.get("strength"),
        }
        for lbl_m, df_m in metric_sheets.items():
            if df_m is None:
                continue
            thresholds = df_m["threshold"].values
            rhos = []
            for t_idx in range(len(thresholds)):
                vals, sd = [], []
                for concept in CONCEPTS:
                    if concept in df_m.columns and concept in semd.index:
                        v = float(df_m[concept].iloc[t_idx])
                        if not np.isnan(v) and v != 0:
                            vals.append(v)
                            sd.append(float(semd.loc[concept, "SemD"]))
                if len(vals) >= 5:
                    r, _ = stats.spearmanr(sd, vals)
                    rhos.append(r)
                else:
                    rhos.append(np.nan)
            ax_c.plot(thresholds, rhos, linewidth=1.5, label=lbl_m,
                      marker=".", markersize=4)
        ax_c.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax_c.set_xlabel("Threshold")
        ax_c.set_ylabel("Spearman ρ (SemD × metric)")
        ax_c.set_title("C. SemD–Metric Correlation Landscape", fontweight="bold")
        ax_c.legend(fontsize=7)
        ax_c.set_ylim(-1, 1)

    # ── D: SemD vs mean |corr| (256-d, current method) ───────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    mean_corrs, semd_d, concept_d = [], [], []
    for concept in CONCEPTS:
        path = corr_dir / f"{concept}_256_{corr_suffix}.csv"
        if path.exists() and concept in semd.index:
            M = np.loadtxt(path, delimiter=",")
            iu = np.triu_indices(M.shape[0], k=1)
            mean_corrs.append(np.mean(np.abs(M[iu])))
            semd_d.append(float(semd.loc[concept, "SemD"]))
            concept_d.append(concept)

    if mean_corrs:
        ax_d.scatter(semd_d, mean_corrs,
                     c=[CONCEPT_COLORS[c] for c in concept_d],
                     s=60, edgecolors="black", linewidths=0.5)
        for i, c in enumerate(concept_d):
            ax_d.annotate(c, (semd_d[i], mean_corrs[i]),
                          fontsize=7, xytext=(3, 3), textcoords="offset points")
        r, p = stats.spearmanr(semd_d, mean_corrs)
        ax_d.set_xlabel("SemD")
        ax_d.set_ylabel(f"Mean |off-diagonal {corr_suffix}| (256-d)")
        ax_d.set_title(f"D. SemD vs Embedding Correlation (ρ={r:.3f}, p={p:.3f})",
                       fontweight="bold")
    else:
        ax_d.set_title("D. SemD vs Embedding Correlation (no data)", fontweight="bold")

    # ── E: SemD vs human network density ─────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    densities, semd_e, concept_e = [], [], []
    for concept in CONCEPTS:
        try:
            M, _ = load_human_network(concept)
            np.fill_diagonal(M, 0)
            n = M.shape[0]
            dens = np.sum(M > 0) / (n * (n - 1)) if n > 1 else 0
            densities.append(dens)
            semd_e.append(float(semd.loc[concept, "SemD"]))
            concept_e.append(concept)
        except Exception:
            pass

    if densities:
        ax_e.scatter(semd_e, densities,
                     c=[CONCEPT_COLORS[c] for c in concept_e],
                     s=60, edgecolors="black", linewidths=0.5)
        for i, c in enumerate(concept_e):
            ax_e.annotate(c, (semd_e[i], densities[i]),
                          fontsize=7, xytext=(3, 3), textcoords="offset points")
        r, p = stats.spearmanr(semd_e, densities)
        ax_e.set_xlabel("SemD")
        ax_e.set_ylabel("Human network density")
        ax_e.set_title(f"E. SemD vs Human Network Density (ρ={r:.3f}, p={p:.3f})",
                       fontweight="bold")

    # ── F: SemD vs BNC frequency ─────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    for concept in CONCEPTS:
        if concept in semd.index:
            ax_f.scatter(float(semd.loc[concept, "lg_BNC_freq"]),
                         float(semd.loc[concept, "SemD"]),
                         color=CONCEPT_COLORS[concept], s=60,
                         edgecolors="black", linewidths=0.5)
            ax_f.annotate(concept,
                          (float(semd.loc[concept, "lg_BNC_freq"]),
                           float(semd.loc[concept, "SemD"])),
                          fontsize=7, xytext=(3, 3), textcoords="offset points")
    r, p = stats.spearmanr(
        semd["lg_BNC_freq"].reindex(CONCEPTS).dropna(),
        semd["SemD"].reindex(CONCEPTS).dropna(),
    )
    ax_f.set_xlabel("log₁₀ BNC Frequency")
    ax_f.set_ylabel("SemD")
    ax_f.set_title(f"F. SemD vs Word Frequency (ρ={r:.3f}, p={p:.3f})",
                   fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = (f"semd_network_integration_{method_name}.png"
                if method_name else "semd_network_integration.png")
    savefig(fig, out_name)


def plot_all_methods_semd_network():
    """Run SemD network integration for every method in METHODS."""
    for method_name, cfg in METHODS.items():
        print(f"  SemD network integration — {method_name}")
        plot_semd_network(
            method_name=method_name,
            results_dir=cfg["results_dir"],
            corr_dir=cfg["corr_dir"],
            corr_suffix=cfg["corr_suffix"],
        )


if __name__ == "__main__":
    plot_all_methods_semd_network()
    print("Done – SemD integration figures.")
