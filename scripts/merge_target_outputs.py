"""Merge several run_target output files, each covering a disjoint subset of
the 10 target_<CODE> columns over the same underlying rows, into a single
file with all target_<CODE> columns present.

Needed when sbatch_run_target.sh was launched in several separate chains
(e.g. targets 1-5 in one chain, 6-10 in another) instead of one continuous
chain of 10 stages: each resulting file only carries the target_<CODE>
columns produced by its own chain. sbatch_run_target_control.sh needs a
single input file carrying all 10 target_<CODE> columns (it reads
target_<CODE> for whichever stage it is running) — this script produces that
file so the control chain can be launched normally afterwards.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse

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
    ap.add_argument("--inputs", nargs="+", required=True, help="run_target output files to merge (>= 2)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--id_col", default="row_id")
    ap.add_argument("--text_col", default="text", help="Used to sanity-check the files share the same rows")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        raise ValueError("Provide at least 2 --inputs to merge.")

    frames = [read_any(p) for p in args.inputs]

    for path, df in zip(args.inputs, frames):
        if args.id_col not in df.columns:
            raise KeyError(f"'{args.id_col}' missing from {path}")

    base_path, base_df = args.inputs[0], frames[0]
    base_ids = base_df[args.id_col]

    for path, df in zip(args.inputs[1:], frames[1:]):
        if len(df) != len(base_df) or not (df[args.id_col].values == base_ids.values).all():
            raise ValueError(
                f"{path} does not line up row-for-row (by '{args.id_col}') with "
                f"{base_path} — these files must come from the same underlying "
                f"corpus, in the same row order, to be merged. Refusing to guess."
            )
        if args.text_col in df.columns and args.text_col in base_df.columns:
            mismatches = int((df[args.text_col].fillna("") != base_df[args.text_col].fillna("")).sum())
            if mismatches:
                print(
                    f"[WARN] {mismatches} row(s) in {path} have a different "
                    f"'{args.text_col}' than {base_path} for the same {args.id_col} — "
                    f"double-check these really are the same corpus.",
                    file=sys.stderr,
                )

    merged = base_df.copy()
    added_from: dict[str, str] = {}

    for path, df in zip(args.inputs[1:], frames[1:]):
        new_cols = [c for c in df.columns if c.startswith("target_")]
        for c in new_cols:
            if c in merged.columns:
                same = merged[c].astype(str).fillna("__NA__").equals(df[c].astype(str).fillna("__NA__"))
                if not same:
                    raise ValueError(
                        f"Column '{c}' is present in both {base_path} and {path} with "
                        f"different values — refusing to silently overwrite. Resolve manually."
                    )
                continue
            merged[c] = df[c].values
            added_from[c] = path

    all_target_cols = sorted(
        {c for df in frames for c in df.columns if c.startswith("target_") and not c.endswith("_JUSTIF")}
    )
    missing = [c for c in all_target_cols if c not in merged.columns]

    print(f"Target columns after merge: {all_target_cols}")
    for c, path in sorted(added_from.items()):
        print(f"  {c} <- {path}")
    if missing:
        print(f"[WARN] still missing after merge: {missing}", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.output.endswith(".csv"):
        merged.to_csv(args.output, index=False)
    else:
        merged.to_parquet(args.output, index=False)

    print(f"Saved: {args.output} ({len(merged):,} rows, {len(merged.columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
