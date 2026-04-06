"""Shared configuration for the gemnet visualization suite."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project")
MATRICES_DIR   = BASE / "gemnet-2/gemini-2/mahalanobis/m_matrices"
CORR_DIR       = BASE / "gemnet-2/gemini-2/mahalanobis/m_corr"
RESULTS_DIR    = BASE / "gemnet-2/gemini-2/mahalanobis/results"
EMBEDDINGS_DIR = BASE / "old_pipeline/take3/embedding_w"
SEMD_PATH      = BASE / "old_pipeline/take4/SemD_data.xlsx"
HUMAN_DIR      = BASE / "human_raw"
HUMAN_CSV_DIR  = HUMAN_DIR / "networks_csv"

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    path = RESULTS_DIR / f"{dim}_mahalanobis_results_master.xlsx"
    if not path.exists():
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
    """Load a Mahalanobis correlation matrix (d×d, headerless CSV)."""
    path = CORR_DIR / f"{concept}_{dim}_mahalanobis.csv"
    return np.loadtxt(path, delimiter=",")
