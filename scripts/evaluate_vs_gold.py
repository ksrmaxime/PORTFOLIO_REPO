# scripts/evaluate_vs_gold.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import pandas as pd

from src.eval import compare_frames


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="LLM output file (csv/parquet)")
    ap.add_argument("--gold", required=True, help="Human-coded reference (csv/parquet)")
    ap.add_argument("--id_col", required=False, help="Join key (e.g. row_uid, sentence_id)")
    ap.add_argument("--cols", required=True, help="Comma-separated columns to compare (e.g. RELEVANT_ART,t,sp,p)")
    ap.add_argument("--keep_na", action="store_true", help="If set, NA pairs are kept (otherwise ignored).")
    ap.add_argument("--use_row_order", action="store_true", help="Compare using row order (creates an internal row_id).")

    args = ap.parse_args()

    pred = read_any(args.pred)
    gold = read_any(args.gold)
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    args.id_col = "__row__"

    if args.use_row_order:
        pred = pred.reset_index(drop=True).copy()
        gold = gold.reset_index(drop=True).copy()

        pred["__row__"] = pred.index.astype(int)
        gold["__row__"] = gold.index.astype(int)

    res = compare_frames(
        pred=pred,
        gold=gold,
        id_col=args.id_col,
        cols=cols,
        drop_na_pairs=(not args.keep_na),
    )

    print("=== EVAL ===")
    print(f"Joined rows: {res.n_total:,}")
    print(f"Compared cells: {res.n_compared:,}")
    print(f"Equal cells: {res.n_equal:,}")
    print(f"Similarity: {res.similarity_pct:.2f}%")

    for c, pct in res.per_column_similarity_pct.items():
        if pct == pct:  # not nan
            print(f"- {c}: {pct:.2f}%")
        else:
            print(f"- {c}: n/a (no comparable values)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())