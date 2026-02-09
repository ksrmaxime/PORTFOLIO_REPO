# scripts/run1_title_triage_batched.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.llm.curnagl_client import TransformersClient, TransformersConfig
from portfolio_repo.llm.run1_title_triage_batched import Run1Config, run1_title_triage_batched


DEFAULT_OUTDIR = "/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/swissdox"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--workdir", required=True)
    p.add_argument("--scratchdir", required=True)

    p.add_argument("--input", default="data/processed/fedlex/laws_structure.parquet")

    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--outname", default="laws_structure_with_title_triage")

    p.add_argument("--model-path", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")

    p.add_argument("--out-col", default="title_ai_relevant")
    p.add_argument("--out-just-col", default="title_ai_justification")

    p.add_argument("--items-per-prompt", type=int, default=40)
    p.add_argument("--prompts-per-batch", type=int, default=8)

    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.0)

    # subset (useful for testing on cluster)
    p.add_argument("--max-rows", type=int, default=None)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    workdir = Path(args.workdir)
    scratchdir = Path(args.scratchdir)

    in_path = (workdir / args.input) if not Path(args.input).is_absolute() else Path(args.input)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_parquet = outdir / f"{args.outname}.parquet"
    out_csv = outdir / f"{args.outname}.csv"

    df = pd.read_parquet(in_path)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    client = TransformersClient(
        TransformersConfig(model_path=args.model_path, dtype=args.dtype, trust_remote_code=True)
    )

    cfg = Run1Config(
        out_col=args.out_col,
        out_just_col=args.out_just_col,
        items_per_prompt=args.items_per_prompt,
        prompts_per_batch=args.prompts_per_batch,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_if_already_done=True,
    )

    df_out = run1_title_triage_batched(client=client, df=df, cfg=cfg)

    df_out.to_parquet(out_parquet, index=False)
    df_out.to_csv(out_csv, index=False)

    triaged = df_out[df_out["level"].isin([1, 2, 3, 4])]
    n_true = int((triaged[cfg.out_col] == True).sum())    # noqa: E712
    n_false = int((triaged[cfg.out_col] == False).sum())  # noqa: E712

    print(f"Input : {in_path}")
    print(f"Saved : {out_parquet}")
    print(f"Saved : {out_csv}")
    print(f"Levels 1..4 classified: TRUE={n_true}, FALSE={n_false}, total={len(triaged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
