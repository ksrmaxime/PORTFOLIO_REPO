from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from portfolio_repo.llm.curnagl_client import TransformersClient, TransformersConfig
from portfolio_repo.llm.run1_title_triage_batched import Run1Config, run1_title_triage_batched

DEFAULT_OUTDIR = "/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    p.add_argument("--scratchdir", required=True)  # kept for compatibility (not used)
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

    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    workdir = Path(a.workdir)

    in_path = (workdir / a.input) if not Path(a.input).is_absolute() else Path(a.input)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_pq = outdir / f"{a.outname}.parquet"
    out_csv = outdir / f"{a.outname}.csv"

    df = pd.read_parquet(in_path)
    if a.max_rows is not None:
        df = df.head(a.max_rows).copy()

    client = TransformersClient(TransformersConfig(model_path=a.model_path, dtype=a.dtype, trust_remote_code=True))
    cfg = Run1Config(
        out_col=a.out_col,
        out_just_col=a.out_just_col,
        items_per_prompt=a.items_per_prompt,
        prompts_per_batch=a.prompts_per_batch,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        skip_if_already_done=True,
    )

    out = run1_title_triage_batched(client, df, cfg)
    out.to_parquet(out_pq, index=False)
    out.to_csv(out_csv, index=False)

    tri = out[out["level"].isin([1, 2, 3, 4])]
    t = int((tri[cfg.out_col] == True).sum())   # noqa: E712
    f = int((tri[cfg.out_col] == False).sum())  # noqa: E712
    print(f"Input : {in_path}")
    print(f"Saved : {out_pq}")
    print(f"Saved : {out_csv}")
    print(f"Levels 1..4 classified: TRUE={t}, FALSE={f}, total={len(tri)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
