# src/run2_config.py
from __future__ import annotations

import pandas as pd


def build_run2_mask(
    df: pd.DataFrame,
    *,
    level_col: str = "level",
    instrument_col: str = "instrument",
) -> pd.Series:
    """
    Run2 processes only:
    - article text rows (level==6)
    - where instrument == True (classified as policy instrument in run1)
    """
    if level_col not in df.columns:
        raise KeyError(f"Missing column: {level_col}")
    if instrument_col not in df.columns:
        raise KeyError(f"Missing column: {instrument_col}")

    levels = pd.to_numeric(df[level_col], errors="coerce")
    instrument = df[instrument_col].astype("boolean")

    return (levels == 6) & (instrument == True)
