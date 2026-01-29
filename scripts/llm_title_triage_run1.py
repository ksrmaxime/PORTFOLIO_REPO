from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from portfolio_repo.llm.client import (
    LLMClient,
    OpenAIHTTPClient,
    OpenAIHTTPConfig,
    VLLMClient,
    VLLMConfig,
)
from portfolio_repo.llm.title_triage import TriageConfig, triage_titles_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--platform", choices=["local", "curnagl"], default="local")

    # paths
    p.add_argument("--workdir", default=None, help="Base directory for repo on cluster (/work/...)")
    p.add_argument("--scratchdir", default=None, help="Base scratch directory (/scratch/<user>/...)")

    p.add_argument("--input", default=None, help="Input parquet path (absolute or relative to workdir)")
    p.add_argument("--output", default=None, help="Output parquet path (absolute or relative to scratchdir/workdir)")

    # LLM
    p.add_argument("--backend", choices=["openai_http", "vllm"], default=None)

    # openai_http backend
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="apertus-local")
    p.add_argument("--timeout", type=int, default=120)

    # vllm backend
    p.add_argument("--model-path", default=None, help="HF model folder, e.g. /reference/LLM/.../...")
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size for vLLM")

    # run params
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out-col", default="title_ai_relevant")

    # subset
    p.add_argument("--law-id", action="append", default=None)
    p.add_argument("--max-laws", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)

    return p.parse_args()


def _default_workdir() -> Path:
    # You can hardcode your /work path here if you want.
    # Otherwise use env var when launching jobs.
    env = os.environ.get("PORTFOLIO_WORKDIR")
    return Path(env) if env else Path.cwd()


def _default_scratchdir() -> Path:
    env = os.environ.get("PORTFOLIO_SCRATCHDIR")
    return Path(env) if env else Path("/tmp")


def _resolve_path(p: str | None, base: Path) -> Path | None:
    if p is None:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def build_client(args: argparse.Namespace) -> LLMClient:
    backend = args.backend
    if backend is None:
        backend = "vllm" if args.platform == "curnagl" else "openai_http"

    if backend == "openai_http":
        return OpenAIHTTPClient(
            OpenAIHTTPConfig(
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
            )
        )

    if backend == "vllm":
        if not args.model_path:
            raise SystemExit(
                "Missing --model-path for vllm backend. "
                "Example: --model-path /reference/LLM/<org>/<model>/"
            )
        return VLLMClient(
            VLLMConfig(
                model_path=args.model_path,
                tensor_parallel_size=args.tp,
            )
        )

    raise SystemExit(f"Unknown backend: {backend}")


def main() -> int:
    args = parse_args()

    workdir = Path(args.workdir) if args.workdir else _default_workdir()
    scratchdir = Path(args.scratchdir) if args.scratchdir else _default_scratchdir()

    # defaults
    default_input = "data/processed/fedlex/laws_structure.parquet"
    default_output_local = "data/processed/fedlex/laws_structure_with_title_triage.parquet"
    default_output_curnagl = "portfolio/run1/laws_structure_with_title_triage.parquet"

    in_path = _resolve_path(args.input or default_input, workdir)

    if args.output:
        # if relative, prefer scratch on curnagl, else workdir on local
        base = scratchdir if (args.platform == "curnagl") else workdir
        out_path = _resolve_path(args.output, base)
    else:
        if args.platform == "curnagl":
            out_path = _resolve_path(default_output_curnagl, scratchdir)
        else:
            out_path = _resolve_path(default_output_local, workdir)

    assert in_path is not None and out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    if args.law_id:
        df = df[df["law_id"].isin(args.law_id)].copy()

    sort_cols = [c for c in ["law_id", "order_index", "level", "node_id"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    if args.max_laws is not None:
        keep = df["law_id"].drop_duplicates().head(args.max_laws).tolist()
        df = df[df["law_id"].isin(keep)].copy()

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    client = build_client(args)

    cfg = TriageConfig(
        chunk_size=args.chunk_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        out_col=args.out_col,
    )

    df_out = triage_titles_dataset(client=client, df=df, cfg=cfg)
    df_out.to_parquet(out_path, index=False)

    non_articles = df_out[df_out["level"] != 5]
    n_true = int((non_articles[cfg.out_col] == True).sum())    # noqa: E712
    n_false = int((non_articles[cfg.out_col] == False).sum())  # noqa: E712

    print(f"Input : {in_path}")
    print(f"Saved : {out_path}")
    print(f"Non-articles classified: TRUE={n_true}, FALSE={n_false}, total={len(non_articles)}")
    print(f"Articles (level=5) left empty: {int((df_out['level']==5).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
