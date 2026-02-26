# src/eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd


@dataclass(frozen=True)
class EvalResult:
    n_total: int
    n_compared: int
    n_equal: int
    similarity_pct: float
    per_column_similarity_pct: Dict[str, float]


def _normalize_series(s: pd.Series) -> pd.Series:
    # normalisation très simple et robuste (labels/strings/bool/int)
    s2 = s.copy()

    # harmoniser NA
    s2 = s2.where(~s2.isna(), other=pd.NA)

    # strings: strip + collapse spaces + upper
    if pd.api.types.is_string_dtype(s2) or s2.dtype == object:
        s2 = s2.astype("string")
        s2 = s2.str.strip()
        s2 = s2.str.replace(r"\s+", " ", regex=True)
        s2 = s2.str.upper()

    # bool: garder bool/NA
    if str(s2.dtype) == "boolean":
        return s2

    return s2


def compare_frames(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    id_col: str,
    cols: List[str],
    drop_na_pairs: bool = True,
) -> EvalResult:
    """
    Compare pred vs gold sur une clé id_col et une liste de colonnes cols.
    similarity_pct = % de cellules identiques (micro-average) sur les colonnes comparées.
    drop_na_pairs=True: ignore les lignes où pred[col] ou gold[col] est NA.
    """
    if id_col not in pred.columns:
        raise ValueError(f"id_col '{id_col}' not found in pred")
    if id_col not in gold.columns:
        raise ValueError(f"id_col '{id_col}' not found in gold")

    # join interne sur id
    p = pred[[id_col] + cols].copy()
    g = gold[[id_col] + cols].copy()

    merged = p.merge(g, on=id_col, how="inner", suffixes=("_pred", "_gold"))

    n_total = len(merged)

    per_col_pct: Dict[str, float] = {}
    n_equal_total = 0
    n_compared_total = 0

    for c in cols:
        sp = _normalize_series(merged[f"{c}_pred"])
        sg = _normalize_series(merged[f"{c}_gold"])

        if drop_na_pairs:
            ok = sp.notna() & sg.notna()
            sp = sp[ok]
            sg = sg[ok]

        n_comp = int(len(sp))
        n_eq = int((sp == sg).sum())

        pct = (100.0 * n_eq / n_comp) if n_comp > 0 else float("nan")
        per_col_pct[c] = pct

        n_equal_total += n_eq
        n_compared_total += n_comp

    similarity_pct = (100.0 * n_equal_total / n_compared_total) if n_compared_total > 0 else float("nan")

    return EvalResult(
        n_total=n_total,
        n_compared=n_compared_total,
        n_equal=n_equal_total,
        similarity_pct=similarity_pct,
        per_column_similarity_pct=per_col_pct,
    )