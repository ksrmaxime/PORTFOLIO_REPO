#!/usr/bin/env python3
"""
Inspect assembled Run 1 prompts before launching the full pipeline.

Usage (from repo root):
    python scripts/inspect_prompts.py
    python scripts/inspect_prompts.py --input data/processed/fedlex/laws_structure_first20laws.parquet
    python scripts/inspect_prompts.py --n 5 --law "Loi fédérale sur..."
    python scripts/inspect_prompts.py --index 42 120 305
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.run1_config import enrich_with_context, build_articles_to_send_mask
from src.run1_prompts import SYSTEM_PROMPT, build_user_prompt

SEPARATOR = "=" * 80


def show_prompt(row: pd.Series, text_col: str, idx: int) -> None:
    print(f"\n{SEPARATOR}")
    print(f"ROW INDEX : {idx}")
    print(f"Law       : {row.get('ctx_law_title', 'N/A')}")
    print(f"Chapter   : {row.get('ctx_chapter_title', 'N/A')}")
    print(f"Article   : {row.get('ctx_article_title', 'N/A')}")
    print(SEPARATOR)
    print("--- SYSTEM PROMPT ---")
    print(SYSTEM_PROMPT)
    print("--- USER PROMPT ---")
    print(build_user_prompt(row, text_col))


def main() -> int:
    ap = argparse.ArgumentParser(description="Preview assembled Run 1 prompts.")
    ap.add_argument(
        "--input",
        default="data/processed/fedlex/all_in_one/laws_structure_final.parquet",
        help="Path to input parquet or CSV.",
    )
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--level_col", default="level")
    ap.add_argument("--label_col", default="label")
    ap.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of random articles to display (ignored if --index is used).",
    )
    ap.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=None,
        help="Specific row indices to display instead of random sampling.",
    )
    ap.add_argument(
        "--law",
        default=None,
        help="Filter: only show articles from a law whose title contains this string.",
    )
    args = ap.parse_args()

    path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path.name}")

    df = enrich_with_context(df, text_col=args.text_col, level_col=args.level_col, label_col=args.label_col)

    mask = build_articles_to_send_mask(df, level_col=args.level_col, text_col=args.text_col)
    articles = df[mask].copy()
    print(f"Articles eligible for Run 1: {len(articles):,}")

    if args.law:
        articles = articles[articles["ctx_law_title"].str.contains(args.law, case=False, na=False)]
        print(f"After law filter '{args.law}': {len(articles):,} articles")

    if len(articles) == 0:
        print("No articles match the criteria.")
        return 0

    if args.index is not None:
        targets = [i for i in args.index if i in articles.index]
        missing = [i for i in args.index if i not in articles.index]
        if missing:
            print(f"Warning: indices not found in eligible articles: {missing}")
        sample = articles.loc[targets]
    else:
        n = min(args.n, len(articles))
        sample = articles.sample(n=n, random_state=42)

    for idx, row in sample.iterrows():
        show_prompt(row, args.text_col, idx)

    print(f"\n{SEPARATOR}")
    print(f"Displayed {len(sample)} prompt(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
