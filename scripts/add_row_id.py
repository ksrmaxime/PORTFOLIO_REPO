"""Add a row_id column (0-based integer) to a parquet or CSV file.

Usage (in-place):
    python scripts/add_row_id.py path/to/file.parquet

Usage (write to separate output):
    python scripts/add_row_id.py path/to/file.parquet --out path/to/output.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def add_row_id(path: Path, col: str = "row_id", overwrite: bool = False, out: Path | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if col in df.columns and not overwrite:
        print(f"Column '{col}' already exists in {path}. Use --overwrite to replace it.")
        return df

    df.insert(0, col, range(len(df)))

    dest = out if out is not None else path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() == ".parquet":
        df.to_parquet(dest, index=False)
    else:
        df.to_csv(dest, index=False)

    print(f"Added '{col}' to {dest} ({len(df):,} rows)")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Add a row_id column to a parquet or CSV file.")
    ap.add_argument("file", help="Path to input parquet or CSV file")
    ap.add_argument("--col", default="row_id", help="Column name to add (default: row_id)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite the column if it already exists")
    ap.add_argument("--out", default=None, help="Output path (default: overwrite input file)")
    args = ap.parse_args()

    add_row_id(
        path=Path(args.file),
        col=args.col,
        overwrite=args.overwrite,
        out=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
