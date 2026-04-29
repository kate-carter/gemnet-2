#converts hpc output to r file 
#auto-generated because I don't need to knwo these packages
"""
Convert HPC det/nondet CSV outputs into R-ready master workbooks.

Expected input naming conventions in a folder:
  - {dim}_det_results.csv
  - {dim}_nondet_results1.csv

Output:
  - {run}_{dim}_results_master.xlsx
with sheets:
  diversity, modularity, corefit, clustering, strength, degree
and each sheet shaped as:
  threshold | banana | bottle | ... | wood
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


SHEET_METRICS = {
    "diversity": "diversity",
    "modularity": "modularity",
    "corefit": "corefit",
    "clustering": "clustering",
    "strength": "strength",
    "degree": "degree",
}

PREFERRED_CONCEPT_ORDER = [
    "banana",
    "bottle",
    "car",
    "chocolate",
    "cookie",
    "grass",
    "key",
    "knife",
    "paper",
    "phone",
    "pickle",
    "pillow",
    "pumpkin",
    "table",
    "wood",
]


def discover_dim_pairs(input_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Return mapping dim -> {'det': path, 'nondet': path} for complete pairs."""
    det_pat = re.compile(r"^(\d+)_det_results\.csv$")
    nd_pat = re.compile(r"^(\d+)_nondet_results1\.csv$")
    found: Dict[str, Dict[str, Path]] = {}

    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        m = det_pat.match(f.name)
        if m:
            found.setdefault(m.group(1), {})["det"] = f
            continue
        m = nd_pat.match(f.name)
        if m:
            found.setdefault(m.group(1), {})["nondet"] = f

    return {dim: paths for dim, paths in found.items() if {"det", "nondet"} <= set(paths)}


def _safe_numeric_threshold(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["threshold"] = pd.to_numeric(out["threshold"], errors="coerce")
    out = out.dropna(subset=["threshold"])
    return out


def _pivot_metric(df: pd.DataFrame, metric_col: str, concept_order: Iterable[str]) -> pd.DataFrame:
    tmp = df[["threshold", "concept", metric_col]].copy()
    pivot = tmp.pivot_table(index="threshold", columns="concept", values=metric_col, aggfunc="mean")
    concept_cols = [c for c in concept_order if c in pivot.columns] + [c for c in pivot.columns if c not in concept_order]
    pivot = pivot.reindex(columns=concept_cols)
    pivot = pivot.sort_index().reset_index()
    pivot.columns.name = None
    return pivot


def build_master(det_csv: Path, nondet_csv: Path) -> Dict[str, pd.DataFrame]:
    det = pd.read_csv(det_csv)
    nondet = pd.read_csv(nondet_csv)

    required_det = {"concept", "threshold", "corefit", "clustering", "strength", "degree"}
    required_nd = {"concept", "threshold", "modularity", "diversity"}
    miss_det = required_det - set(det.columns)
    miss_nd = required_nd - set(nondet.columns)
    if miss_det:
        raise ValueError(f"{det_csv.name} missing columns: {sorted(miss_det)}")
    if miss_nd:
        raise ValueError(f"{nondet_csv.name} missing columns: {sorted(miss_nd)}")

    det = _safe_numeric_threshold(det)
    nondet = _safe_numeric_threshold(nondet)

    merged = pd.merge(
        det[["concept", "threshold", "corefit", "clustering", "strength", "degree"]],
        nondet[["concept", "threshold", "modularity", "diversity"]],
        on=["concept", "threshold"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(f"No overlapping concept+threshold rows between {det_csv.name} and {nondet_csv.name}")

    sheets: Dict[str, pd.DataFrame] = {}
    for sheet_name, metric_col in SHEET_METRICS.items():
        sheets[sheet_name] = _pivot_metric(merged, metric_col, PREFERRED_CONCEPT_ORDER)
    return sheets


def write_master(sheets: Dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in ["diversity", "modularity", "corefit", "clustering", "strength", "degree"]:
            sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)


def infer_run_from_folder(input_dir: Path) -> str:
    """
    Infer run prefix from folder name.
    Example:
      me_results -> me
      mg_results -> mg
      take4_results -> take4
    """
    name = input_dir.name
    if "_results" in name:
        return name.split("_results", 1)[0]
    # fallback: use full folder name if no suffix found
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HPC det/nondet outputs to master workbook format.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/gemnet-2/gemini-2/mahalanobis/me_results"),
        help="Folder containing {dim}_det_results.csv and {dim}_nondet_results1.csv files.",
    )
    parser.add_argument(
        "--dim",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of dimensions to process (e.g., --dim 256 768 1536). If omitted, process all discovered pairs.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pairs = discover_dim_pairs(input_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No matching pairs found in {input_dir}. Need both *_det_results.csv and *_nondet_results1.csv."
        )

    run = infer_run_from_folder(input_dir)
    print(f"[info] run inferred from folder '{input_dir.name}' -> '{run}'")

    dims = args.dim if args.dim else sorted(pairs.keys(), key=lambda x: int(x))
    for dim in dims:
        if dim not in pairs:
            print(f"[skip] {dim}: missing det/nondet pair in {input_dir}")
            continue
        det_csv = pairs[dim]["det"]
        nd_csv = pairs[dim]["nondet"]
        out = output_dir / f"{run}_{dim}_results_master.xlsx"
        sheets = build_master(det_csv, nd_csv)
        write_master(sheets, out)
        print(f"[ok] wrote {out.name} from {det_csv.name} + {nd_csv.name}")


if __name__ == "__main__":
    main()