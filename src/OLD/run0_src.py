from __future__ import annotations

import unicodedata
from typing import Iterable

import pandas as pd


def _strip_accents(s: str) -> str:
    # Convert "données" -> "donnees", etc.
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def build_ai_relevant_column(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    level_col: str = "level",
    relevant_levels: Iterable[int] = (1, 2, 3, 4, 5),
    keywords: Iterable[str] = ("données", "registre", "système", "automatisé"),
) -> pd.Series:
    """
    Returns a nullable boolean Series:
      - True/False for rows where level in relevant_levels
      - <NA> otherwise (e.g., level 0 or 6)
    """
    if level_col not in df.columns:
        raise KeyError(f"Missing column: {level_col}")
    if label_col not in df.columns:
        raise KeyError(f"Missing column: {label_col}")

    levels = pd.to_numeric(df[level_col], errors="coerce")
    in_scope = levels.isin(list(relevant_levels))

    # label text (safe)
    label = df[label_col].fillna("").astype(str)

    # Two-pass matching: with accents + accent-stripped (robust)
    label_lc = label.str.casefold()
    label_lc_noacc = label.map(_strip_accents).str.casefold()

    kw = [str(k) for k in keywords]
    kw_lc = [k.casefold() for k in kw]
    kw_noacc = [_strip_accents(k).casefold() for k in kw]

    hit = pd.Series(False, index=df.index)
    for k1, k2 in zip(kw_lc, kw_noacc):
        hit = hit | label_lc.str.contains(k1, regex=False) | label_lc_noacc.str.contains(k2, regex=False)

    # Build output: boolean for in-scope, NA elsewhere
    out = pd.Series(pd.NA, index=df.index, dtype="boolean")
    out.loc[in_scope] = hit.loc[in_scope].astype("boolean")

    return out