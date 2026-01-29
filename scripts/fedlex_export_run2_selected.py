# scripts/fedlex_export_run2_selected.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.extract.selected_articles_export import (
    SelectedArticlesExportConfig,
    export_selected_articles,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        help="Path to parquet result (e.g. data/processed/...parquet).",
    )
    p.add_argument(
        "--out-csv",
        default="data/processed/exports/run2_selected_articles.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--out-md",
        default=None,
        help="Optional output Markdown path (nice for printing).",
    )

    # column overrides if needed
    p.add_argument("--selected-col", default="run2_selected")
    p.add_argument("--blocks-col", default="article_target_blocks")
    p.add_argument("--justif-col", default="article_justification")
    p.add_argument("--label-col", default="label")
    p.add_argument("--text-col", default="text")
    p.add_argument("--law-id-col", default="law_id")
    p.add_argument("--level-col", default="level")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.input)

    cfg = SelectedArticlesExportConfig(
        selected_col=args.selected_col,
        blocks_col=args.blocks_col,
        justification_col=args.justif_col,
        label_col=args.label_col,
        text_col=args.text_col,
        law_id_col=args.law_id_col,
        level_col=args.level_col,
    )

    out_csv = Path(args.out_csv)
    export_selected_articles(df=df, out_path=out_csv, cfg=cfg, fmt="csv")
    print(f"Saved CSV: {out_csv}")

    if args.out_md:
        out_md = Path(args.out_md)
        export_selected_articles(df=df, out_path=out_md, cfg=cfg, fmt="md")
        print(f"Saved MD: {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
