from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_repo.llm.client import LLMConfig  # type: ignore
from portfolio_repo.llm.run3_instruments import Run3Config, run3_extract_instruments


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/processed/fedlex/laws_structure_with_article_classification.parquet",
        help="Parquet input (run2 output).",
    )
    ap.add_argument(
        "--output",
        default="data/processed/fedlex/laws_structure_with_article_classification_and_instruments.parquet",
        help="Parquet output (run3).",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="apertus-local")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--process-all", action="store_true", help="Retraiter même si article_instruments déjà rempli.")
    args = ap.parse_args()

    cfg = Run3Config(
        input_path=Path(args.input),
        output_path=Path(args.output),
        checkpoint_every=args.checkpoint_every,
        sleep_seconds=args.sleep_seconds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        only_missing=not args.process_all,
    )

    llm_cfg = LLMConfig(base_url=args.base_url, model=args.model, timeout=args.timeout)
    return run3_extract_instruments(cfg, llm_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
