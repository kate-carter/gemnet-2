"""Shared configuration for the gemnet visualization suite."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project")
GEMINI2_DIR    = BASE / "gemnet-2/gemini-2"
EMBEDDINGS_DIR = BASE / "old_pipeline/take3/embedding_w"
SEMD_PATH      = BASE / "old_pipeline/take4/SemD_data.xlsx"
HUMAN_DIR      = BASE / "human_raw"
HUMAN_CSV_DIR  = HUMAN_DIR / "networks_csv"
# Canonical human BCT statistics (from PI's published dataset).
# Sheet 'human_bct': rows = metrics (hum_corefit, hum_modularity, ...),
#                    columns = concept names.
HUMAN_STATS_PATH = BASE / "gemnet-2/gemini-2/human_stats.xlsx"

# ── Per-method directory registry ──────────────────────────────────────────────
# Used by multi-method scripts (vis_human_correlation, vis_method_comparison).
# Each entry provides all paths and style properties for one correlation method.
METHODS = {
    "Mah. Gaussian": {
        "results_dir":  GEMINI2_DIR / "mahalanobis/mg_results",
        "corr_dir":     GEMINI2_DIR / "mahalanobis/mg_corr",
        "matrices_dir": GEMINI2_DIR / "mahalanobis/mg_matrices",
        "prefix":       "mg",
        "corr_suffix":  "mahalanobis",
        "color":        "#e41a1c",
        "marker":       "o",
    },
    "Mah. Euclidean": {
        "results_dir":  GEMINI2_DIR / "mahalanobis/me_results",
        "corr_dir":     GEMINI2_DIR / "mahalanobis/me_corr",
        "matrices_dir": GEMINI2_DIR / "mahalanobis/me_matrices",
        "prefix":       "me",
        "corr_suffix":  "mahalanobis",
        "color":        "#984ea3",
        "marker":       "v",
    },
    "Spearman": {
        "results_dir":  GEMINI2_DIR / "spearman/sp_results",
        "corr_dir":     GEMINI2_DIR / "spearman/sp_corr",
        "matrices_dir": GEMINI2_DIR / "spearman/sp_matrices",
        "prefix":       "sp",
        "corr_suffix":  "spearman",
        "color":        "#377eb8",
        "marker":       "s",
    },
    "Pearson": {
        "results_dir":  GEMINI2_DIR / "pearson/pe_results",
        "corr_dir":     GEMINI2_DIR / "pearson/pe_corr",
        "matrices_dir": GEMINI2_DIR / "pearson/pe_matrices",
        "prefix":       "pe",
        "corr_suffix":  "pearson",
        "color":        "#4daf4a",
        "marker":       "^",
    },
    "Cosine": {
        "results_dir":  GEMINI2_DIR / "cosine/cos_results",
        "corr_dir":     GEMINI2_DIR / "cosine/cos_corr",
        "matrices_dir": GEMINI2_DIR / "cosine/cos_matrices",
        "prefix":       "cos",
        "corr_suffix":  "cosine",
        "color":        "#ff7f00",
        "marker":       "D",
    },
}

# ── Single-method shortcuts (Spearman) ────────────────────────────────────────
# vis_correlation_landscape, vis_network_gallery, vis_semd_network, and
# vis_threshold_sweep operate on Spearman data by default; they use these
# aliases so they don't need to be changed if you want to point them elsewhere.
MATRICES_DIR = METHODS["Spearman"]["matrices_dir"]
CORR_DIR     = METHODS["Spearman"]["corr_dir"]
RESULTS_DIR  = METHODS["Spearman"]["results_dir"]

OUTPUT_DIR = GEMINI2_DIR / "vis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Concepts ───────────────────────────────────────────────────────────────────
CONCEPTS = [
    "banana", "bottle", "car", "chocolate", "cookie",
    "grass", "key", "knife", "paper", "phone",
    "pickle", "pillow", "pumpkin", "table", "wood",
]

DIMS = [256, 512, 768, 1536, 3072]
THRESHOLDS = np.round(np.arange(0.05, 0.95, 0.05), 2)

# ── Color palettes ─────────────────────────────────────────────────────────────
CONCEPT_CMAP = plt.cm.tab20
CONCEPT_COLORS = {c: CONCEPT_CMAP(i / len(CONCEPTS)) for i, c in enumerate(CONCEPTS)}

DIM_COLORS = {
    256:  "#1b9e77",
    512:  "#d95f02",
    768:  "#7570b3",
    1536: "#e7298a",
    3072: "#66a61e",
}

# ── Styling ────────────────────────────────────────────────────────────────────
STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

def apply_style():
    mpl.rcParams.update(STYLE)

def savefig(fig, name, **kw):
    path = OUTPUT_DIR / name
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


def load_results_workbook(dim):
    """Load a results master xlsx, returning dict of sheet_name → DataFrame."""
    import pandas as pd
    path_candidates = [
        RESULTS_DIR / f"sp_{dim}_results_master.xlsx",
        RESULTS_DIR / f"sp_{dim}_master_results.xlsx",
        RESULTS_DIR / f"{dim}_master_results.xlsx",
        RESULTS_DIR / f"{dim}_results_master.xlsx",
    ]
    path = next((p for p in path_candidates if p.exists()), None)
    if path is None:
        return None
    xls = pd.ExcelFile(path, engine="openpyxl")
    return {s: xls.parse(s) for s in xls.sheet_names}


def load_human_network(concept):
    """Load a human network CSV as a numpy array + list of node labels."""
    import pandas as pd
    path = HUMAN_CSV_DIR / f"{concept}_human_network.csv"
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(float), list(df.columns)


def load_corr_matrix(concept, dim):
    """Load a Spearman correlation matrix (d×d, headerless CSV)."""
    path = CORR_DIR / f"{concept}_{dim}_spearman.csv"
    return np.loadtxt(path, delimiter=",")
