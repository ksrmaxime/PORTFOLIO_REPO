from __future__ import annotations

import pandas as pd


def enrich_with_context(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "label",
    level_col: str = "level",
) -> pd.DataFrame:
    """
    For each level-6 row, adds three context columns by scanning backward:
      ctx_law_title     — label of the nearest preceding level-0 row
      ctx_chapter_title — label of the nearest preceding level 1–4 row
      ctx_article_title — label of the nearest preceding level-5 row
    All other rows receive empty strings in these columns.
    """
    if level_col not in df.columns:
        raise KeyError(f"Missing column: {level_col}")
    if label_col not in df.columns:
        raise KeyError(f"Missing column: {label_col}")

    df = df.copy()
    levels = pd.to_numeric(df[level_col], errors="coerce")
    labels = df[label_col].fillna("").astype(str)

    ctx_law: list[str] = []
    ctx_chapter: list[str] = []
    ctx_article: list[str] = []

    cur_law = ""
    cur_chapter = ""
    cur_article = ""

    for i in df.index:
        lvl = levels.at[i]
        if lvl == 0:
            cur_law = labels.at[i]
            cur_chapter = ""
            cur_article = ""
        elif lvl in (1, 2, 3, 4):
            cur_chapter = labels.at[i]
            cur_article = ""
        elif lvl == 5:
            cur_article = labels.at[i]

        ctx_law.append(cur_law)
        ctx_chapter.append(cur_chapter)
        ctx_article.append(cur_article)

    df["ctx_law_title"] = ctx_law
    df["ctx_chapter_title"] = ctx_chapter
    df["ctx_article_title"] = ctx_article

    return df


def build_articles_to_send_mask(
    df: pd.DataFrame,
    *,
    level_col: str = "level",
    text_col: str = "text",
    relevance_col: str | None = None,  # kept for backward compat, unused
) -> pd.Series:
    """
    Returns a boolean mask selecting level-6 rows that have non-empty text.
    Rows with empty or NaN text (abrogated articles) are excluded.
    """
    if level_col not in df.columns:
        raise KeyError(f"Missing column: {level_col}")

    levels = pd.to_numeric(df[level_col], errors="coerce")
    is_article = levels == 6

    if text_col in df.columns:
        has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
        return is_article & has_text

    return is_article
