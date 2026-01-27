from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.llm.client import LLMConfig, LocalLLMClient
from portfolio_repo.llm.article_run2 import Run2Config, classify_selected_articles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/processed/fedlex/laws_structure_with_title_triage.parquet",
        help="Output of run 1 (must include title_ai_relevant + row_uid).",
    )
    p.add_argument(
        "--output",
        default="data/processed/fedlex/laws_structure_with_article_classification.parquet",
        help="Output parquet path.",
    )

    # LLM
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="apertus-local")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=900)

    # subset / test
    p.add_argument("--law-id", action="append", default=None)
    p.add_argument("--max-laws", type=int, default=None)
    p.add_argument("--max-selected-articles", type=int, default=None)

    # columns
    p.add_argument("--title-col", default="title_ai_relevant")
    p.add_argument("--selected-col", default="run2_selected")
    p.add_argument("--out-ai-col", default="article_ai_relevant")
    p.add_argument("--out-targets-col", default="article_targets")
    p.add_argument("--out-justif-col", default="article_justification")

    # resume behavior
    p.add_argument("--no-skip", action="store_true", help="Recompute even if already classified.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    # subset filtering for quick tests
    if args.law_id:
        df = df[df["law_id"].isin(args.law_id)].copy()

    # keep first N laws
    if args.max_laws is not None:
        law_ids = df["law_id"].drop_duplicates().head(args.max_laws).tolist()
        df = df[df["law_id"].isin(law_ids)].copy()

    client = LocalLLMClient(
        LLMConfig(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
        )
    )

    cfg = Run2Config(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        out_selected_col=args.selected_col,
        out_ai_relevant_col=args.out_ai_col,
        out_targets_col=args.out_targets_col,
        out_justification_col=args.out_justif_col,
        skip_if_already_done=(not args.no_skip),
    )

    df_out = classify_selected_articles(
        client=client,
        df=df,
        cfg=cfg,
        title_relevant_col=args.title_col,
    )

    # Optional: cap number of classified articles for testing (after selection)
    if args.max_selected_articles is not None:
        # Keep dataset intact, but blank out results beyond N selected articles
        sel = (df_out["level"] == 5) & (df_out[cfg.out_selected_col] == True)  # noqa: E712
        sel_idx = df_out[sel].index.tolist()
        keep_idx = set(sel_idx[: args.max_selected_articles])
        drop_idx = [i for i in sel_idx if i not in keep_idx]
        for col in [cfg.out_ai_relevant_col, cfg.out_targets_col, cfg.out_justification_col]:
            df_out.loc[drop_idx, col] = pd.NA

    df_out.to_parquet(out_path, index=False)

    # logs
    sel_articles = df_out[(df_out["level"] == 5) & (df_out[cfg.out_selected_col] == True)]  # noqa: E712
    done = sel_articles[cfg.out_ai_relevant_col].notna().sum()
    n_true = int((sel_articles[cfg.out_ai_relevant_col] == True).sum())  # noqa: E712
    n_false = int((sel_articles[cfg.out_ai_relevant_col] == False).sum())  # noqa: E712

    print(f"Saved: {out_path}")
    print(f"Selected articles: {len(sel_articles)}")
    print(f"Classified (non-null): {int(done)} | TRUE={n_true} FALSE={n_false}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
