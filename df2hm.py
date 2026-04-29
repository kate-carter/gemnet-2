"""
Convert corrvar CSV outputs to Excel (.xlsx) with light cleaning.

Edit the CONFIG section at the top for future naming conventions.
"""

from pathlib import Path

import pandas as pd


# =========================
# CONFIG: naming conventions
# =========================
INPUT_DIR = Path(
    "/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/gemnet-2/gemini-2/r_results"
)
OUTPUT_DIR = INPUT_DIR

# Direct file mode (simple, explicit)
INPUT_FILENAME = "dimvar_sp_cor_5.csv"
OUTPUT_FILENAME = "dimvar_sp_cor_5.xlsx"

# Optional pattern mode (set True to batch-convert matching files)
USE_PATTERN_MODE = False
INPUT_GLOB = "*dimvar_sp_cor*.csv"
CSV_SUFFIX_TO_REPLACE = ".csv"
XLSX_SUFFIX = ".xlsx"

SHEET_NAME = "heatmap"

# Common R-export index column names to drop
DROP_INDEX_COLUMN_NAMES = {"", "Unnamed: 0", "X", "row.names", "Row Labels"}

# If True, trim row count so threshold length matches the
# longest correlation column (all non-threshold/non-index columns).
TRIM_TO_LONGEST_CORRELATION = True

# Output shape:
# - True  => long format: threshold, method, correlation
# - False => wide format (original columns)
OUTPUT_LONG_FORMAT = True

# In long format, keep rows where correlation is NA.
# Set False if you prefer to drop missing correlation rows.
KEEP_NA_CORRELATIONS = False


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight cleanup so Excel output matches analysis-ready format."""
    out = df.copy()

    # Drop auto-generated row index column from CSV exports (if present)
    first_col = str(out.columns[0]).strip() if len(out.columns) else ""
    if first_col in DROP_INDEX_COLUMN_NAMES:
        out = out.drop(columns=[out.columns[0]])

    # Keep threshold first if it exists
    if "threshold" in out.columns:
        ordered_cols = ["threshold"] + [c for c in out.columns if c != "threshold"]
        out = out[ordered_cols]

    # Optional: align row count to the longest correlation column length.
    # This ensures threshold is trimmed BEFORE stacking to long format.
    if TRIM_TO_LONGEST_CORRELATION and "threshold" in out.columns:
        corr_cols = [c for c in out.columns if c != "threshold"]
        if corr_cols:
            lengths = [int(out[col].notna().sum()) for col in corr_cols]
            keep_rows = max(lengths) if lengths else len(out)
            out = out.iloc[:keep_rows].reset_index(drop=True)

    return out


def _convert_one(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, na_values=["NA", "NaN", "nan"])
    cleaned = _clean_dataframe(df)
    if "threshold" not in cleaned.columns:
        raise ValueError("Expected a 'threshold' column after cleaning.")

    if OUTPUT_LONG_FORMAT:
        corr_cols = [c for c in cleaned.columns if c != "threshold"]
        converted = cleaned.melt(
            id_vars="threshold",
            value_vars=corr_cols,
            var_name="dimensionality",
            value_name="correlation",
        )
        if not KEEP_NA_CORRELATIONS:
            converted = converted[converted["correlation"].notna()].copy()
    else:
        converted = cleaned

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_excel(output_path, index=False, sheet_name=SHEET_NAME)
    print(f"Converted: {input_path} -> {output_path}")


def main() -> None:
    if USE_PATTERN_MODE:
        matches = sorted(INPUT_DIR.glob(INPUT_GLOB))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {INPUT_DIR / INPUT_GLOB}")
        for input_path in matches:
            out_name = input_path.name.replace(CSV_SUFFIX_TO_REPLACE, XLSX_SUFFIX)
            _convert_one(input_path, OUTPUT_DIR / out_name)
    else:
        input_path = INPUT_DIR / INPUT_FILENAME
        output_path = OUTPUT_DIR / OUTPUT_FILENAME
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        _convert_one(input_path, output_path)
    print(INPUT_FILENAME, "converted to", OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
