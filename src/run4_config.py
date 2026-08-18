# src/run4_config.py
from __future__ import annotations

import pandas as pd


def build_run4_mask(
    df: pd.DataFrame,
    *,
    ai_relevant_col: str = "AI_RELEVANT",
) -> pd.Series:
    """
    Run4 only audits rows where run3 classified AI_RELEVANT as True.
    Most run3 errors are false positives (too eager to classify True),
    so rows classified False by run3 are not sent to run4.
    """
    if ai_relevant_col not in df.columns:
        raise KeyError(f"Missing column: {ai_relevant_col}")

    return df[ai_relevant_col].eq(True)
