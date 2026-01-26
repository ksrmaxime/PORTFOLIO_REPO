from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.structure.build_structure import build_structure_parquet_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a structure-level parquet (Partie/Titre/Chapitre/Section/Art) from laws parquet."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input laws parquet (columns: base_act_uri, title, clean_text, ...).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output structure parquet.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    laws = pd.read_parquet(in_path)
    structure_df = build_structure_parquet_df(laws)

    structure_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(structure_df):,} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
