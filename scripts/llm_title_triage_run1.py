# scripts/llm_title_triage_run1.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.llm.client import LLMConfig, LocalLLMClient
from portfolio_repo.llm.title_triage import TriageConfig, triage_titles_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/fedlex/laws_structure.parquet")
    p.add_argument("--output", default="data/processed/fedlex/laws_structure_with_title_triage.parquet")

    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="apertus-local")
    p.add_argument("--timeout", type=int, default=120)

    p.add_argument("--out-col", default="title_ai_relevant")

    # NEW: run subset
    p.add_argument(
        "--law-id",
        action="append",
        default=None,
        help="Filter to a specific law_id. Repeatable: --law-id X --law-id Y",
    )
    p.add_argument(
        "--max-laws",
        type=int,
        default=None,
        help="Keep only the first N distinct law_id (after sorting).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Keep only the first N rows (after sorting).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    # subset filtering (for quick tests)
    if args.law_id:
        df = df[df["law_id"].isin(args.law_id)].copy()

    # stable order for deterministic subset selection
    sort_cols = [c for c in ["law_id", "order_index", "level", "node_id"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    if args.max_laws is not None:
        keep = df["law_id"].drop_duplicates().head(args.max_laws).tolist()
        df = df[df["law_id"].isin(keep)].copy()

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    client = LocalLLMClient(
        LLMConfig(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
        )
    )

    cfg = TriageConfig(
        chunk_size=args.chunk_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        out_col=args.out_col,
    )

    df_out = triage_titles_dataset(client=client, df=df, cfg=cfg)
    df_out.to_parquet(out_path, index=False)

    non_articles = df_out[df_out["level"] != 5]
    n_true = int((non_articles[cfg.out_col] == True).sum())   # noqa: E712
    n_false = int((non_articles[cfg.out_col] == False).sum()) # noqa: E712
    print(f"Saved: {out_path}")
    print(f"Non-articles classified: TRUE={n_true}, FALSE={n_false}, total={len(non_articles)}")
    print(f"Articles (level=5) left empty: {int((df_out['level']==5).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

