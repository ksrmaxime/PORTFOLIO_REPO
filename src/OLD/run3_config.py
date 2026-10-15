# src/run3_config.py
from __future__ import annotations

import pandas as pd


def build_run3_mask(
    df: pd.DataFrame,
    *,
    instrument_col: str = "instrument",
    instrument_confirmed_col: str = "INSTRUMENT_CONFIRMED",
) -> pd.Series:
    """
    Run3 processes articles confirmed as true policy instruments:
      - instrument==True  AND INSTRUMENT_CONFIRMED==True  (run1 said YES, run2 confirmed)
      - instrument==False AND INSTRUMENT_CONFIRMED==False (run1 said NO,  run2 overturned → true instrument)
    """
    if instrument_col not in df.columns:
        raise KeyError(f"Missing column: {instrument_col}")
    if instrument_confirmed_col not in df.columns:
        raise KeyError(f"Missing column: {instrument_confirmed_col}")

    instrument = df[instrument_col].astype("boolean")
    confirmed = df[instrument_confirmed_col].astype("boolean")

    return (instrument.eq(True) & confirmed.eq(True)) | (instrument.eq(False) & confirmed.eq(False))
