from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from portfolio.ai_relevant_filter import build_ai_relevant_column


# =========================
# EDIT ONLY THESE TWO LINES
# =========================
INPUT_PARQUET = "/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure_selected.parquet"
OUTPUT_DIR = "/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed"
# =========================


LABEL_COL = "label"
LEVEL_COL = "level"
NEW_COL = "AI_RELEVANT"

KEYWORDS = ("données", "registre", "système", "automatisé")
RELEVANT_LEVELS = (1, 2, 3, 4, 5)


def main() -> int:
    in_path = Path(INPUT_PARQUET)
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
    stem = in_path.stem
    out_path = out_dir / f"{stem}_with_{NEW_COL.lower()}_job{job_id}.parquet"

    df = pd.read_parquet(in_path)

    # Add column (nullable boolean)
    df[NEW_COL] = build_ai_relevant_column(
        df,
        label_col=LABEL_COL,
        level_col=LEVEL_COL,
        relevant_levels=RELEVANT_LEVELS,
        keywords=KEYWORDS,
    )

    df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote: {out_path}")
    print(f"[INFO] Rows total: {len(df):,}")
    print(f"[INFO] In-scope levels (1-5): {pd.to_numeric(df[LEVEL_COL], errors='coerce').isin(RELEVANT_LEVELS).sum():,}")
    print(f"[INFO] True count (among in-scope): {df[NEW_COL].sum(skipna=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())