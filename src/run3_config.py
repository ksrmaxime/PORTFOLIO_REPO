# src/run3_config.py
from __future__ import annotations

import pandas as pd


def build_run3_mask(
    df: pd.DataFrame,
    *,
    instrument_confirmed_col: str = "INSTRUMENT_CONFIRMED",
) -> pd.Series:
    """
    Run3 processes only rows where INSTRUMENT_CONFIRMED == True
    (articles confirmed as policy instruments by run2).
    """
    if instrument_confirmed_col not in df.columns:
        raise KeyError(f"Missing column: {instrument_confirmed_col}")

    return df[instrument_confirmed_col].eq(True)
