#!/usr/bin/env python3
"""
Figure 4 – SemD (Semantic Diversity) integration with network properties.

Panels:
  A  SemD values for the 15 concepts (ranked bar chart)
  B  SemD vs network metric at best-match threshold (scatter + regression)
  C  Correlation landscape: Spearman r (SemD × metric) across thresholds
  D  SemD vs mean absolute correlation (embedding geometry)
  E  SemD vs human network density
  F  SemD × BNC frequency, colored by concept category
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
    load_human_network,
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


def plot_semd_network():
    semd = load_semd()

    results_path = RESULTS_DIR / "256_mahalanobis_results_master.xlsx"
    has_results = results_path.exists()
    if has_results:
        xls = pd.ExcelFile(results_path, engine="openpyxl")
        sheets = {s: xls.parse(s) for s in xls.sheet_names}

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    fig.suptitle(
        "Semantic Diversity × Network Structure",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ── A: SemD ranked bar chart ───────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ordered = semd.sort_values("SemD", ascending=True)
    y = range(len(ordered))
    colors = [CONCEPT_COLORS[c] for c in ordered.index]
    ax_a.barh(y, ordered["SemD"].values, color=colors, edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(ordered.index, fontsize=9)
    ax_a.set_xlabel("Semantic Diversity (SemD)")
    ax_a.set_title("A. SemD Values for 15 Concepts", fontweight="bold")

    # ── B: SemD vs modularity at a representative threshold ────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if has_results:
        df_mod = sheets.get("modularity18")
        if df_mod is None:
            df_mod = sheets.get("modularity")
        if df_mod is not None:
            t_idx = 2  # threshold ~0.15 (moderate sparsity)
            threshold_val = df_mod["threshold"].iloc[t_idx]
            mod_vals, semd_vals, concept_list = [], [], []
            for concept in CONCEPTS:
                if concept in df_mod.columns and concept in semd.index:
                    mod_vals.append(float(df_mod[concept].iloc[t_idx]))
                    semd_vals.append(float(semd.loc[concept, "SemD"]))
                    concept_list.append(concept)

            ax_b.scatter(
                semd_vals, mod_vals,
                c=[CONCEPT_COLORS[c] for c in concept_list],
                s=60, edgecolors="black", linewidths=0.5, zorder=5,
            )
            for i, c in enumerate(concept_list):
                ax_b.annotate(c, (semd_vals[i], mod_vals[i]),
                              fontsize=7, ha="left", va="bottom",
                              xytext=(3, 3), textcoords="offset points")

            r, p = stats.spearmanr(semd_vals, mod_vals)
            ax_b.set_xlabel("SemD")
            ax_b.set_ylabel(f"Modularity (t={threshold_val:.2f})")
            ax_b.set_title(f"B. SemD vs Modularity (ρ={r:.3f}, p={p:.3f})",
                           fontweight="bold")

    # ── C: Correlation landscape: Spearman(SemD, metric) × threshold ──────
    ax_c = fig.add_subplot(gs[0, 2])
    if has_results:
        metric_sheets = {
            "Diversity": sheets["diversity18"] if "diversity18" in sheets else sheets.get("diversity"),
            "Modularity": sheets["modularity18"] if "modularity18" in sheets else sheets.get("modularity"),
            "Clustering": sheets.get("clustering"),
            "Core Fit": sheets.get("corefit"),
            "Strength": sheets.get("strength"),
        }
        for label, df_m in metric_sheets.items():
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
            ax_c.plot(thresholds, rhos, linewidth=1.5, label=label, marker=".", markersize=4)
        ax_c.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax_c.set_xlabel("Threshold")
        ax_c.set_ylabel("Spearman ρ (SemD × metric)")
        ax_c.set_title("C. SemD–Metric Correlation Landscape", fontweight="bold")
        ax_c.legend(fontsize=7)
        ax_c.set_ylim(-1, 1)

    # ── D: SemD vs mean absolute correlation (256-d) ──────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    mean_corrs, semd_d, concept_d = [], [], []
    for concept in CONCEPTS:
        path = CORR_DIR / f"{concept}_256_mahalanobis.csv"
        if path.exists() and concept in semd.index:
            M = np.loadtxt(path, delimiter=",")
            iu = np.triu_indices(M.shape[0], k=1)
            mean_corrs.append(np.mean(np.abs(M[iu])))
            semd_d.append(float(semd.loc[concept, "SemD"]))
            concept_d.append(concept)

    if mean_corrs:
        ax_d.scatter(
            semd_d, mean_corrs,
            c=[CONCEPT_COLORS[c] for c in concept_d],
            s=60, edgecolors="black", linewidths=0.5,
        )
        for i, c in enumerate(concept_d):
            ax_d.annotate(c, (semd_d[i], mean_corrs[i]),
                          fontsize=7, xytext=(3, 3), textcoords="offset points")
        r, p = stats.spearmanr(semd_d, mean_corrs)
        ax_d.set_xlabel("SemD")
        ax_d.set_ylabel("Mean |off-diagonal correlation| (256-d)")
        ax_d.set_title(f"D. SemD vs Embedding Correlation (ρ={r:.3f}, p={p:.3f})",
                       fontweight="bold")

    # ── E: SemD vs human network density ──────────────────────────────────
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
        ax_e.scatter(
            semd_e, densities,
            c=[CONCEPT_COLORS[c] for c in concept_e],
            s=60, edgecolors="black", linewidths=0.5,
        )
        for i, c in enumerate(concept_e):
            ax_e.annotate(c, (semd_e[i], densities[i]),
                          fontsize=7, xytext=(3, 3), textcoords="offset points")
        r, p = stats.spearmanr(semd_e, densities)
        ax_e.set_xlabel("SemD")
        ax_e.set_ylabel("Human network density")
        ax_e.set_title(f"E. SemD vs Human Network Density (ρ={r:.3f}, p={p:.3f})",
                       fontweight="bold")

    # ── F: SemD vs BNC frequency ──────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    for concept in CONCEPTS:
        if concept in semd.index:
            ax_f.scatter(
                float(semd.loc[concept, "lg_BNC_freq"]),
                float(semd.loc[concept, "SemD"]),
                color=CONCEPT_COLORS[concept], s=60,
                edgecolors="black", linewidths=0.5,
            )
            ax_f.annotate(
                concept,
                (float(semd.loc[concept, "lg_BNC_freq"]),
                 float(semd.loc[concept, "SemD"])),
                fontsize=7, xytext=(3, 3), textcoords="offset points",
            )
    r, p = stats.spearmanr(
        semd["lg_BNC_freq"].reindex(CONCEPTS).dropna(),
        semd["SemD"].reindex(CONCEPTS).dropna(),
    )
    ax_f.set_xlabel("log₁₀ BNC Frequency")
    ax_f.set_ylabel("SemD")
    ax_f.set_title(f"F. SemD vs Word Frequency (ρ={r:.3f}, p={p:.3f})", fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "semd_network_integration.png")


if __name__ == "__main__":
    plot_semd_network()
    print("Done – SemD integration figures.")
