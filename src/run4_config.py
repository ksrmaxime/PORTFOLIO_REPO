# src/run4_config.py
from __future__ import annotations

import pandas as pd


def build_run4_mask(
    df: pd.DataFrame,
    *,
    ai_relevant_col: str = "AI_RELEVANT",
) -> pd.Series:
    """
    Run4 processes all rows where AI_RELEVANT is not NA
    (all articles classified by run3, both True and False).
    """
    if ai_relevant_col not in df.columns:
        raise KeyError(f"Missing column: {ai_relevant_col}")

    return df[ai_relevant_col].notna()
