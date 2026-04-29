"""
vis_structural_drivers.py — Structural Drivers of Network Variation
====================================================================

Adapted from structure_driversm.py (test-pipeline) for the gemnet-2
multi-method pipeline.  One 6-panel figure is generated per correlation
method (Mahalanobis, Spearman, Pearson, Cosine).

  Panel A:  PCA of Structural Profiles — concepts as points in BCT-metric
            space (mean across all available thresholds × dimensionalities).
            Reveals the actual axes of structural variation.

  Panel B:  Metric Co-variation — Spearman ρ matrix of BCT metrics across
            concepts.  Shows which structural properties share a common axis
            vs. vary independently.

  Panel C:  Edge Weight Distribution Properties — skewness, kurtosis,
            entropy, and positive-fraction of the 256-d correlation matrices
            per concept.

  Panel D:  BCT Metrics as Predictors of SemD — Spearman ρ (and p-value)
            of every BCT metric (concept-level mean) against SemD.
            (* = p < .05).  Modularity and Corefit are highlighted.

  Panel E:  Best Predictor Deep Dive — scatter of the strongest BCT
            predictor of SemD with a regression line.

  Panel F:  Structural Clustering Dendrogram — Ward hierarchical clustering
            of concepts by their full BCT profiles.

Output: vis_results/structural_drivers_{Method}.png  (one file per method)
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, skew, kurtosis
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

    class StandardScaler:
        """Minimal numpy drop-in for sklearn.preprocessing.StandardScaler."""
        def fit_transform(self, X):
            self._mean = X.mean(axis=0)
            self._std  = X.std(axis=0)
            self._std[self._std == 0] = 1.0
            return (X - self._mean) / self._std

    class PCA:
        """Minimal numpy drop-in for sklearn.decomposition.PCA."""
        def __init__(self, n_components=None):
            self.n_components = n_components

        def fit_transform(self, X):
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            k = self.n_components if self.n_components else X.shape[1]
            k = min(k, S.shape[0])
            total_var = (S ** 2).sum()
            self.explained_variance_ratio_ = (S[:k] ** 2) / total_var
            self.components_ = Vt[:k]
            return U[:, :k] * S[:k]

from vis_config import (
    apply_style, savefig, OUTPUT_DIR,
    CONCEPTS, CONCEPT_COLORS, DIMS, METHODS,
    SEMD_PATH,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

PRIORITY_METRICS = ["modularity", "corefit"]
HIGHLIGHT_COLOR  = "#d62728"   # red highlight for priority metrics in Panel D


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_workbook(results_dir, prefix, dim):
    """Return path to the master results workbook for one dim, or None."""
    candidates = [
        results_dir / f"{prefix}_{dim}_results_master.xlsx",
        results_dir / f"{prefix}_{dim}_master_results.xlsx",
        results_dir / f"{dim}_results_master.xlsx",
        results_dir / f"{dim}_master_results.xlsx",
    ]
    return next((p for p in candidates if p.exists()), None)


def _load_bct_sheets(results_dir, prefix, dim):
    """Load all sheets from one master workbook → {metric: DataFrame}."""
    path = _find_workbook(results_dir, prefix, dim)
    if path is None:
        return {}
    xls = pd.ExcelFile(path, engine="openpyxl")
    out = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if df.empty or len(df.columns) < 2:
            continue
        if "threshold" not in df.columns:
            df = df.rename(columns={df.columns[0]: "threshold"})
        out[sheet.strip().lower()] = df
    return out


def load_bct_profiles(results_dir, prefix):
    """Build a per-concept profile dataframe (mean BCT across all dims/thresholds).

    Returns DataFrame with 'concept' column + one column per BCT metric.
    Also returns the raw per-(dim, metric, threshold) data for Panel D.
    """
    all_records = []   # list of {concept, metric, value} over all dims/thresholds

    for dim in DIMS:
        sheets = _load_bct_sheets(results_dir, prefix, dim)
        for metric, df in sheets.items():
            for _, row in df.iterrows():
                for concept in CONCEPTS:
                    if concept in df.columns:
                        val = row[concept]
                        if pd.notna(val):
                            all_records.append(
                                {"concept": concept, "metric": metric,
                                 "dim": dim, "value": float(val)}
                            )

    if not all_records:
        return pd.DataFrame(columns=["concept"]), pd.DataFrame()

    raw_df = pd.DataFrame(all_records)

    # Mean per concept × metric
    agg = (raw_df.groupby(["concept", "metric"])["value"]
           .mean().reset_index()
           .pivot(index="concept", columns="metric", values="value")
           .reset_index())
    agg.columns.name = None
    return agg, raw_df


def load_semd(concepts):
    """Load SemD values from SEMD_PATH."""
    if not SEMD_PATH.exists():
        return {}
    try:
        df = pd.read_excel(SEMD_PATH, skiprows=1, header=None,
                           names=["term", "mean_cos", "SemD",
                                  "BNC_wordcount", "BNC_contexts",
                                  "BNC_freq", "lg_BNC_freq"])
        df["term"] = df["term"].astype(str).str.strip().str.lower()
        return {c: float(df.loc[df["term"] == c.lower(), "SemD"].values[0])
                for c in concepts
                if c.lower() in df["term"].values}
    except Exception as exc:
        print(f"    SemD load error: {exc}")
        return {}


def load_corr_matrices(corr_dir, concepts, dim=256):
    """Load full (unthresholded) correlation matrices for each concept at dim."""
    mats = {}
    for c in concepts:
        # Try multiple suffix conventions
        suffix_candidates = [c, f"{c}_w"]
        for sc in suffix_candidates:
            for fname_try in [
                f"{sc}_{dim}_spearman.csv",
                f"{sc}_{dim}_mahalanobis.csv",
                f"{sc}_{dim}_cosine.csv",
                f"{sc}_{dim}_pearson.csv",
                f"{sc}_{dim}_corr.csv",
            ]:
                path = corr_dir / fname_try
                if path.exists():
                    try:
                        arr = pd.read_csv(path, index_col=0).values.astype(float)
                        mats[c] = arr
                    except Exception:
                        pass
                    break
            if c in mats:
                break
    return mats


def edge_weight_features(mat):
    """Distributional features of the upper-triangle edge weights."""
    n = mat.shape[0]
    upper = mat[np.triu_indices(n, k=1)]
    pos = upper[upper > 0]
    hist, _ = np.histogram(upper, bins=50, density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0
    return {
        "skewness":     float(skew(upper)),
        "kurtosis":     float(kurtosis(upper)),
        "entropy":      entropy,
        "pos_fraction": float(len(pos) / len(upper)) if len(upper) > 0 else 0.0,
    }


# ── Main figure ───────────────────────────────────────────────────────────────

def _plot_structural_drivers(method_name, cfg):
    """Build the 6-panel figure for one correlation method."""
    results_dir  = cfg["results_dir"]
    corr_dir     = cfg["corr_dir"]
    prefix       = cfg["prefix"]
    method_color = cfg["color"]

    print(f"  [{method_name}] Loading BCT profiles …")
    profiles_df, _ = load_bct_profiles(results_dir, prefix)

    if profiles_df.empty or len(profiles_df) < 2:
        print(f"    No data found — skipping {method_name}.")
        return

    # Restrict to concepts present in profiles
    concepts = [c for c in CONCEPTS if c in profiles_df["concept"].values]
    profiles_df = profiles_df[profiles_df["concept"].isin(concepts)].copy()
    profiles_df = profiles_df.set_index("concept").reindex(concepts).reset_index()

    metric_cols = [c for c in profiles_df.columns
                   if c != "concept" and profiles_df[c].notna().sum() >= 3]

    print(f"    {len(concepts)} concepts, {len(metric_cols)} metrics: {metric_cols}")

    # Load SemD
    semd = load_semd(concepts)
    print(f"    SemD: {len(semd)} concepts matched")

    # Load corr matrices for Panel C (256-d)
    print(f"  [{method_name}] Loading corr matrices (256d) for edge-weight features …")
    corr_mats = load_corr_matrices(corr_dir, concepts, dim=256)
    print(f"    Loaded {len(corr_mats)}/{len(concepts)} corr matrices")

    cc = CONCEPT_COLORS  # concept → color

    # ── Figure layout ─────────────────────────────────────────────────────────
    apply_style()
    fig = plt.figure(figsize=(28, 24))
    gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.30,
                           left=0.07, right=0.95, top=0.92, bottom=0.05)

    # ── Panel A: PCA of structural profiles ───────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    X = profiles_df[metric_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(5, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    proj = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    for c_name, x, y in zip(profiles_df["concept"], proj[:, 0], proj[:, 1]):
        col = cc.get(c_name, "#888888")
        ax_a.scatter(x, y, c=[col], s=90, edgecolors="black", linewidth=0.5,
                     zorder=3)
        ax_a.annotate(c_name, (x, y), fontsize=7.5, ha="center", va="bottom",
                      xytext=(0, 5), textcoords="offset points", color=col)

    ax_a.set_xlabel(f"Structural PC1 ({var_exp[0]:.1%})", fontsize=12)
    ax_a.set_ylabel(f"Structural PC2 ({var_exp[1]:.1%})", fontsize=12)
    ax_a.set_title("A.  PCA of Structural Profiles",
                   fontsize=13, fontweight="bold", loc="left")
    ax_a.grid(True, alpha=0.2)

    loadings = pca.components_[:2, :]
    top_pc1 = [metric_cols[i] for i in np.argsort(np.abs(loadings[0]))[-3:]]
    top_pc2 = [metric_cols[i] for i in np.argsort(np.abs(loadings[1]))[-3:]]
    ax_a.text(0.02, 0.02,
              f"PC1: {', '.join(top_pc1)}\nPC2: {', '.join(top_pc2)}",
              transform=ax_a.transAxes, fontsize=8, va="bottom",
              bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # ── Panel B: Metric co-variation ──────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    if len(metric_cols) > 1:
        corr_df = profiles_df[metric_cols].corr(method="spearman")
        labels = [c.replace("_", " ") for c in metric_cols]
        fs = 7 if len(labels) > 10 else 8
        im_b = ax_b.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1,
                           interpolation="nearest")
        ax_b.set_xticks(range(len(labels)))
        ax_b.set_xticklabels(labels, fontsize=fs, rotation=45, ha="right")
        ax_b.set_yticks(range(len(labels)))
        ax_b.set_yticklabels(labels, fontsize=fs)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = corr_df.values[i, j]
                ax_b.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=max(5, fs - 1),
                          color="white" if abs(v) > 0.6 else "black")
        fig.colorbar(im_b, ax=ax_b, shrink=0.8, label="Spearman ρ")

    ax_b.set_title("B.  Metric Co-variation (Spearman ρ, mean across dims)",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Panel C: Edge weight distribution properties ──────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])

    ew_feats = ["skewness", "kurtosis", "entropy", "pos_fraction"]
    ew_data = {}
    for c in concepts:
        if c in corr_mats:
            ew = edge_weight_features(corr_mats[c])
            for k in ew_feats:
                ew_data.setdefault(k, {})[c] = ew[k]

    available_ew = [f for f in ew_feats if f in ew_data and len(ew_data[f]) > 0]
    if available_ew:
        x_pos = np.arange(len(concepts))
        width = 0.8 / len(available_ew)
        for i, col in enumerate(available_ew):
            vals = np.array([ew_data[col].get(c, np.nan) for c in concepts],
                            dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                continue
            normed = vals - np.nanmin(finite)
            mx = np.nanmax(normed)
            if mx > 0:
                normed = normed / mx
            colors = [cc.get(c, "#888888") for c in concepts]
            ax_c.bar(x_pos + i * width, normed, width, color=colors,
                     alpha=0.55 + 0.15 * i, edgecolor="white",
                     linewidth=0.3, label=col)

        ax_c.set_xticks(x_pos + width * len(available_ew) / 2)
        ax_c.set_xticklabels(concepts, fontsize=7, rotation=45, ha="right")
        for lbl in ax_c.get_xticklabels():
            lbl.set_color(cc.get(lbl.get_text(), "black"))
        ax_c.legend(fontsize=9, loc="upper right")

    ax_c.set_ylabel("Normalised Value (256d)", fontsize=12)
    ax_c.set_title("C.  Edge Weight Distribution Properties (256d corr matrices)",
                   fontsize=13, fontweight="bold", loc="left")
    ax_c.grid(True, axis="y", alpha=0.2)

    # ── Panel D: BCT metrics as predictors of SemD ────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    semd_results = {}   # metric → (ρ, p)
    if len(semd) >= 5:
        semd_arr = np.array([semd.get(c, np.nan) for c in concepts])
        for metric in metric_cols:
            bct_arr = profiles_df.set_index("concept").reindex(concepts)[metric].values.astype(float)
            mask = np.isfinite(bct_arr) & np.isfinite(semd_arr)
            if mask.sum() >= 5:
                r, p = spearmanr(bct_arr[mask], semd_arr[mask])
                semd_results[metric] = (r, p)

    if semd_results:
        sorted_metrics = sorted(semd_results.keys(),
                                key=lambda m: abs(semd_results[m][0]),
                                reverse=True)
        rho_vals = [semd_results[m][0] for m in sorted_metrics]
        p_vals   = [semd_results[m][1] for m in sorted_metrics]
        y_pos = np.arange(len(sorted_metrics))

        bar_colors = [
            HIGHLIGHT_COLOR if m in PRIORITY_METRICS else method_color
            for m in sorted_metrics
        ]
        bars = ax_d.barh(y_pos, rho_vals, color=bar_colors, alpha=0.75,
                         edgecolor="black", linewidth=0.4)

        # p-value stars
        for yi, (m, r, p) in enumerate(
                zip(sorted_metrics, rho_vals, p_vals)):
            if p < 0.05:
                x_ann = r + (0.02 if r >= 0 else -0.02)
                ha_ann = "left" if r >= 0 else "right"
                ax_d.text(x_ann, yi, "*", fontsize=12,
                          va="center", ha=ha_ann, color="black")

        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels([m.replace("_", " ") for m in sorted_metrics],
                              fontsize=8)
        ax_d.axvline(0, color="black", linewidth=0.8)
        ax_d.set_xlabel("Spearman ρ with SemD", fontsize=11)
        ax_d.set_xlim(-1, 1)
        ax_d.grid(True, axis="x", alpha=0.2)

        # Legend patch for priority metrics
        import matplotlib.patches as mpatches
        priority_patch = mpatches.Patch(color=HIGHLIGHT_COLOR,
                                        label="Priority (modularity / corefit)")
        other_patch    = mpatches.Patch(color=method_color, label="Other metric")
        ax_d.legend(handles=[priority_patch, other_patch], fontsize=8,
                    loc="lower right")
    else:
        ax_d.text(0.5, 0.5, "SemD not available",
                  ha="center", va="center", transform=ax_d.transAxes, fontsize=12)

    ax_d.set_title("D.  BCT Metrics as Predictors of SemD  (* p < .05)",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Panel E: Best predictor scatter ───────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 0])

    if semd_results:
        best_metric = max(semd_results, key=lambda m: abs(semd_results[m][0]))
        best_r, best_p = semd_results[best_metric]

        bct_arr = profiles_df.set_index("concept").reindex(concepts)[best_metric].values.astype(float)
        semd_arr = np.array([semd.get(c, np.nan) for c in concepts])

        xs, ys, cs_plot = [], [], []
        for c, bv, sv in zip(concepts, bct_arr, semd_arr):
            if np.isfinite(bv) and np.isfinite(sv):
                xs.append(bv)
                ys.append(sv)
                cs_plot.append(cc.get(c, "#888888"))
                ax_e.scatter(bv, sv, c=[cc.get(c, "#888888")], s=80,
                             edgecolors="black", linewidth=0.5, zorder=3)
                ax_e.annotate(c, (bv, sv), fontsize=7.5, ha="center",
                              va="bottom", xytext=(0, 5),
                              textcoords="offset points",
                              color=cc.get(c, "#888888"))

        if len(xs) > 2:
            z = np.polyfit(xs, ys, 1)
            x_line = np.linspace(min(xs), max(xs), 50)
            ax_e.plot(x_line, np.polyval(z, x_line), "--",
                      color="grey", alpha=0.6, linewidth=2)

        ax_e.set_xlabel(best_metric.replace("_", " ").title(), fontsize=12)
        ax_e.set_ylabel("SemD", fontsize=12)
        ax_e.text(0.03, 0.97,
                  f"Best predictor: {best_metric.replace('_', ' ')}\n"
                  f"Spearman ρ = {best_r:.3f}, p = {best_p:.4f}",
                  transform=ax_e.transAxes, fontsize=10, va="top",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax_e.grid(True, alpha=0.2)
    else:
        ax_e.text(0.5, 0.5, "SemD not available",
                  ha="center", va="center", transform=ax_e.transAxes, fontsize=12)

    ax_e.set_title("E.  Best Structural Predictor of SemD",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Panel F: Structural clustering dendrogram ─────────────────────────────
    ax_f = fig.add_subplot(gs[2, 1])

    X_clean = profiles_df[metric_cols].fillna(0).values
    if X_clean.shape[0] >= 2 and X_clean.shape[1] >= 1:
        X_norm = StandardScaler().fit_transform(X_clean)
        dist   = pdist(X_norm, metric="euclidean")
        Z      = linkage(dist, method="ward")

        dend = dendrogram(Z, labels=list(profiles_df["concept"]),
                          ax=ax_f, leaf_rotation=45, leaf_font_size=8,
                          above_threshold_color="#cccccc")
        for lbl in ax_f.get_xticklabels():
            c_name = lbl.get_text()
            lbl.set_color(cc.get(c_name, "black"))
            lbl.set_fontweight("bold")

    ax_f.set_ylabel("Ward Distance", fontsize=12)
    ax_f.set_title("F.  Structural Clustering of Concepts (Ward, BCT profiles)",
                   fontsize=13, fontweight="bold", loc="left")

    # ── Suptitle & legend ─────────────────────────────────────────────────────
    fig.suptitle(
        f"Structural Drivers of Network Variation — {method_name} "
        f"(mean across {len(DIMS)} dimensionalities)",
        fontsize=18, fontweight="bold", y=1.005,
    )

    out_name = f"structural_drivers_{method_name}.png"
    savefig(fig, out_name, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")


def plot_structural_drivers():
    """Entry point: generate one figure per correlation method."""
    apply_style()
    for method_name, cfg in METHODS.items():
        print(f"\n  Structural drivers — {method_name}")
        try:
            _plot_structural_drivers(method_name, cfg)
        except Exception as exc:
            import traceback
            print(f"    ERROR for {method_name}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    plot_structural_drivers()
