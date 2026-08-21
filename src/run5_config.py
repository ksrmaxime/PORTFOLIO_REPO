# src/run5_config.py
from __future__ import annotations

import pandas as pd


def build_run5_mask(
    df: pd.DataFrame,
    *,
    final_ai_relevant_col: str = "final_ai_relevant",
) -> pd.Series:
    """
    Run5 only codes articles that survived the full relevance pipeline
    (run1 instrument detection -> run2 audit -> run3 AI-relevance ->
    run4 audit), i.e. final_ai_relevant == True in the run4 output.
    """
    if final_ai_relevant_col not in df.columns:
        raise KeyError(f"Missing column: {final_ai_relevant_col}")

    return df[final_ai_relevant_col].astype("boolean").eq(True)
