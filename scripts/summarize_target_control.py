"""Summarize the output of the last run_target_control stage (HIGH_STAKES_RISKS,
which carries all 10 control_target_<CODE> columns) into a recap of the
articles still classified True after control, and for which target(s).

Two files are produced:
  - <output_base>_wide.csv / .parquet : one row per article still True after
    control, with a control_target_<CODE> column per target plus a
    'confirmed_targets' summary column (semicolon-joined codes) and
    'n_confirmed_targets'.
  - <output_base>_long.csv : one row per (article, confirmed target) pair,
    with the corresponding control_target_<CODE>_JUSTIF text when available —
    convenient for manual review / pivoting.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet/CSV output of the last run_target_control stage (control_target_HIGH_STAKES_RISKS)")
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--level_col", default="level")
    ap.add_argument(
        "--id_cols",
        nargs="+",
        default=["row_id", "law_id", "label"],
        help="Columns to carry through to identify each article (kept only if present in the input)",
    )
    args = ap.parse_args()

    df = read_any(args.input)

    control_cols = [
        c for c in df.columns if c.startswith("control_target_") and not c.endswith("_JUSTIF")
    ]
    if not control_cols:
        print("No control_target_<CODE> columns found in input — nothing to summarize.", file=sys.stderr)
        return 1

    if args.level_col in df.columns:
        levels = pd.to_numeric(df[args.level_col], errors="coerce")
        is_article = levels == 6
    else:
        is_article = pd.Series(True, index=df.index)

    confirmed_mask = is_article & df[control_cols].eq(True).any(axis=1)

    id_cols = [c for c in args.id_cols if c in df.columns]

    wide = df.loc[confirmed_mask, id_cols + control_cols].copy()
    codes = [c[len("control_target_"):] for c in control_cols]
    wide["confirmed_targets"] = [
        ";".join(code for code, val in zip(codes, row) if val is True)
        for row in wide[control_cols].itertuples(index=False)
    ]
    wide["n_confirmed_targets"] = wide[control_cols].eq(True).sum(axis=1)

    Path(args.output_base).parent.mkdir(parents=True, exist_ok=True)
    wide_parquet = f"{args.output_base}_wide.parquet"
    wide_csv = f"{args.output_base}_wide.csv"
    wide.to_parquet(wide_parquet, index=False)
    wide.to_csv(wide_csv, index=False)

    long_rows = []
    for control_col, code in zip(control_cols, codes):
        justif_col = f"{control_col}_JUSTIF"
        sub_mask = confirmed_mask & df[control_col].eq(True)
        sub = df.loc[sub_mask, id_cols].copy()
        sub["target"] = code
        sub["justification"] = df.loc[sub_mask, justif_col] if justif_col in df.columns else pd.NA
        long_rows.append(sub)

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(columns=id_cols + ["target", "justification"])
    long_csv = f"{args.output_base}_long.csv"
    long_df.to_csv(long_csv, index=False)

    print(f"Articles considered (level == 6 if present): {int(is_article.sum()):,}")
    print(f"Articles still True for >=1 target after control: {int(confirmed_mask.sum()):,}")
    for control_col, code in zip(control_cols, codes):
        n = int((confirmed_mask & df[control_col].eq(True)).sum())
        print(f"  {code}: {n:,}")
    print(f"Saved: {wide_parquet}")
    print(f"Saved: {wide_csv}")
    print(f"Saved: {long_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
