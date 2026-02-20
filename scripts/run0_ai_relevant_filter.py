from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from portfolio_repo.ai_relevant_filter import build_ai_relevant_column


# =========================
# EDIT ONLY THESE TWO LINES
# =========================
INPUT_PARQUET = "/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure.parquet"
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

    out_parquet = out_dir / f"{stem}_with_{NEW_COL.lower()}_job{job_id}.parquet"
    out_csv = out_dir / f"{stem}_with_{NEW_COL.lower()}_job{job_id}.csv"

    df = pd.read_parquet(in_path)

    # Add column (nullable boolean)
    df[NEW_COL] = build_ai_relevant_column(
        df,
        label_col=LABEL_COL,
        level_col=LEVEL_COL,
        relevant_levels=RELEVANT_LEVELS,
        keywords=KEYWORDS,
    )

    # 1) Parquet (keeps nullable boolean cleanly)
    df.to_parquet(out_parquet, index=False)

    # 2) CSV (for manual checking)
    # Convert pandas BooleanDtype to something CSV-friendly:
    # True/False/"" (empty for NA)
    df_csv = df.copy()
    df_csv[NEW_COL] = df_csv[NEW_COL].map(lambda x: "" if pd.isna(x) else bool(x))

    df_csv.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[OK] Wrote parquet: {out_parquet}")
    print(f"[OK] Wrote csv:    {out_csv}")
    print(f"[INFO] Rows total: {len(df):,}")
    in_scope = pd.to_numeric(df[LEVEL_COL], errors="coerce").isin(RELEVANT_LEVELS)
    print(f"[INFO] In-scope levels (1-5): {in_scope.sum():,}")
    print(f"[INFO] True count (among in-scope): {df.loc[in_scope, NEW_COL].sum(skipna=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())