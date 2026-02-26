from __future__ import annotations

import pandas as pd


def _get_relevance_col(df: pd.DataFrame) -> str:
    # Tolérant : tu as parlé de RELEVANT_AI, mais run1 créait AI_RELEVANT
    for c in ("RELEVANT_AI", "AI_RELEVANT"):
        if c in df.columns:
            return c
    raise KeyError("Missing relevance column: expected RELEVANT_AI or AI_RELEVANT")


def build_articles_to_send_mask(
    df: pd.DataFrame,
    *,
    level_col: str = "level",
    relevance_col: str | None = None,
) -> pd.Series:
    """
    Returns boolean mask over df.index for rows (level==6) that must be sent to LLM.

    Rules implemented:
    A) If a section/subchapter (level in 1..4) is marked True, then send ALL level==6 rows
       until the next level in 1..4 (regardless of level==5 True/False inside).
    B) If an article name row (level==5) is marked True but not inside an active True section,
       then send ONLY the immediately following row if it is level==6.
    """
    if level_col not in df.columns:
        raise KeyError(f"Missing column: {level_col}")

    rel_col = relevance_col or _get_relevance_col(df)

    levels = pd.to_numeric(df[level_col], errors="coerce")
    rel = df[rel_col]

    # Normalize relevance to pandas nullable boolean
    rel_bool = rel.astype("boolean")

    n = len(df)
    send = pd.Series(False, index=df.index)

    active_true_section = False

    # Iterate in original order (assumes parquet preserves structural order)
    # If you have a specific ordering column, sort before calling this.
    for i in range(n):
        lvl = levels.iat[i]
        r = rel_bool.iat[i]

        # Section boundaries: any level 1..4 row resets section state
        if lvl in (1, 2, 3, 4):
            active_true_section = bool(r) if pd.notna(r) else False
            continue

        # If inside a True section, send all article-text rows (level 6)
        if active_true_section:
            if lvl == 6:
                send.iat[i] = True
            continue

        # Outside True section: special case level 5 marked True -> send next row if level 6
        if lvl == 5 and pd.notna(r) and bool(r) is True:
            if i + 1 < n and levels.iat[i + 1] == 6:
                send.iat[i + 1] = True

    return send