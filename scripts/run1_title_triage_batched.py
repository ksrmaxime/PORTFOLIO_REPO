# scripts/run1_title_triage_batched.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.llm.curnagl_client import TransformersClient, TransformersConfig
from portfolio_repo.llm.run1_title_triage_batched import Run1Config, run1_title_triage_batched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--workdir", required=True)
    p.add_argument("--scratchdir", required=True)

    p.add_argument("--input", default="data/processed/fedlex/laws_structure.parquet")
    p.add_argument("--output", default="portfolio/run1/laws_structure_with_title_triage.parquet")

    p.add_argument("--model-path", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")

    p.add_argument("--out-col", default="title_ai_relevant")

    p.add_argument("--items-per-prompt", type=int, default=40)
    p.add_argument("--prompts-per-batch", type=int, default=8)

    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.0)

    # subset (useful for testing on cluster)
    p.add_argument("--max-rows", type=int, default=None)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    workdir = Path(args.workdir)
    scratchdir = Path(args.scratchdir)

    in_path = (workdir / args.input) if not Path(args.input).is_absolute() else Path(args.input)
    out_path = (scratchdir / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    client = TransformersClient(
        TransformersConfig(model_path=args.model_path, dtype=args.dtype, trust_remote_code=True)
    )

    cfg = Run1Config(
        out_col=args.out_col,
        items_per_prompt=args.items_per_prompt,
        prompts_per_batch=args.prompts_per_batch,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_if_already_done=True,
    )

    df_out = run1_title_triage_batched(client=client, df=df, cfg=cfg)
    df_out.to_parquet(out_path, index=False)

    non_articles = df_out[df_out["level"].isin([1, 2, 3, 4])]
    n_true = int((non_articles[cfg.out_col] == True).sum())    # noqa: E712
    n_false = int((non_articles[cfg.out_col] == False).sum())  # noqa: E712

    print(f"Input : {in_path}")
    print(f"Saved : {out_path}")
    print(f"Non-articles classified: TRUE={n_true}, FALSE={n_false}, total={len(non_articles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
