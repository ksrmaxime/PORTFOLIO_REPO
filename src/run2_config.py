# src/run3_config.py
from __future__ import annotations

import pandas as pd


def build_run3_mask(
    df: pd.DataFrame,
    *,
    level_col: str = "level",
    relevant_art_col: str = "RELEVANT_ART",
) -> pd.Series:
    """
    Run3 processes only:
    - article text rows (level==6)
    - where RELEVANT_ART == True (from run2)

    Returns boolean mask over df.index.
    """
    if level_col not in df.columns:
        raise KeyError(f"Missing column: {level_col}")
    if relevant_art_col not in df.columns:
        raise KeyError(f"Missing column: {relevant_art_col}")

    levels = pd.to_numeric(df[level_col], errors="coerce")
    rel = df[relevant_art_col].astype("boolean")

    return (levels == 6) & (rel == True)