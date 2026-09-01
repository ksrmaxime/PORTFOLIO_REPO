"""Filter a run_target output parquet down to articles relevant to at least
one of the 10 targets.

Handoff point between the target-classification chain (sbatch_run_target.sh,
10 stages) and the instrument-classification chain (sbatch_run_inst.sh, 7
stages): running instrument prompts on the full AI-relevant corpus is
wasteful since only a small fraction of articles end up target-relevant,
while essentially every article carries at least one instrument. Filtering
here first keeps the instrument stage cheap.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet output of the last run_target stage")
    ap.add_argument("--output", required=True)
    ap.add_argument("--level_col", default="level")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    target_cols = [
        c for c in df.columns if c.startswith("target_") and not c.endswith("_JUSTIF")
    ]
    if not target_cols:
        print("No target_<CODE> columns found in input — nothing to filter.", file=sys.stderr)
        return 1

    if args.level_col in df.columns:
        levels = pd.to_numeric(df[args.level_col], errors="coerce")
        is_article = levels == 6
    else:
        is_article = pd.Series(True, index=df.index)

    any_target_true = df[target_cols].eq(True).any(axis=1)
    keep = is_article & any_target_true

    out = df.loc[keep].reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    print(f"Target columns used: {target_cols}")
    print(f"Articles (level==6): {int(is_article.sum()):,}")
    print(f"Target-relevant (>=1 target True): {int(keep.sum()):,}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
