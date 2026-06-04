# src/run3_config.py
from __future__ import annotations

import pandas as pd


def build_run3_mask(
    df: pd.DataFrame,
    *,
    ai_relevant_col: str = "AI_RELEVANT",
) -> pd.Series:
    """
    Run3 processes only rows where AI_RELEVANT == True
    (articles classified as AI-relevant by run2).
    """
    if ai_relevant_col not in df.columns:
        raise KeyError(f"Missing column: {ai_relevant_col}")

    return df[ai_relevant_col].eq(True)
