# src/portfolio_repo/extract/selected_articles_export.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd


@dataclass(frozen=True)
class SelectedArticlesExportConfig:
    selected_col: str = "run2_selected"
    level_col: str = "level"
    law_id_col: str = "law_id"

    label_col: str = "label"
    text_col: str = "text"

    blocks_col: str = "article_target_blocks"
    justification_col: str = "article_justification"

    # law title = label where level==0
    law_title_level: int = 0
    article_level: int = 5


def _bool_true_series(s: pd.Series) -> pd.Series:
    # Parquet peut contenir bool, object, ou NA; on force True strict
    return s.fillna(False).astype(bool)


def build_selected_articles_table(
    df: pd.DataFrame, cfg: SelectedArticlesExportConfig
) -> pd.DataFrame:
    """
    Returns a flat table:
    law_id | law_title | article_label | text | article_target_blocks | article_justification
    only for selected articles (run2_selected==True and level==5).
    """
    required = [
        cfg.law_id_col,
        cfg.level_col,
        cfg.label_col,
        cfg.text_col,
        cfg.selected_col,
        cfg.blocks_col,
        cfg.justification_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df2 = df.copy()

    # Law titles
    law_titles = (
        df2[df2[cfg.level_col] == cfg.law_title_level]
        .groupby(cfg.law_id_col, sort=False)[cfg.label_col]
        .first()
        .rename("law_title")
    )

    # Selected articles
    sel_mask = (df2[cfg.level_col] == cfg.article_level) & _bool_true_series(df2[cfg.selected_col])
    articles = df2.loc[sel_mask, [cfg.law_id_col, cfg.label_col, cfg.text_col, cfg.blocks_col, cfg.justification_col]].copy()

    articles = articles.rename(
        columns={
            cfg.label_col: "article_label",
            cfg.text_col: "text",
            cfg.blocks_col: "article_target_blocks",
            cfg.justification_col: "article_justification",
        }
    )

    # Attach law title
    articles = articles.merge(
        law_titles.reset_index(),
        how="left",
        left_on=cfg.law_id_col,
        right_on=cfg.law_id_col,
    )

    # Reorder columns
    articles = articles[
        [cfg.law_id_col, "law_title", "article_label", "text", "article_target_blocks", "article_justification"]
    ]

    # Sort for readability (stable)
    if "order_index" in df2.columns:
        # re-merge order_index just for sorting
        sort_keys = df2.loc[sel_mask, [cfg.law_id_col, "order_index"]].copy()
        sort_keys = sort_keys.reset_index().rename(columns={"index": "_idx"})
        articles = articles.reset_index().rename(columns={"index": "_idx"})
        articles = articles.merge(sort_keys, on=["_idx", cfg.law_id_col], how="left").sort_values(
            [cfg.law_id_col, "order_index"], kind="stable"
        )
        articles = articles.drop(columns=["_idx", "order_index"]).reset_index(drop=True)
    else:
        articles = articles.sort_values([cfg.law_id_col], kind="stable").reset_index(drop=True)

    return articles


def export_selected_articles(
    df: pd.DataFrame,
    out_path: Path,
    cfg: SelectedArticlesExportConfig,
    fmt: Literal["csv", "md"] = "csv",
    encoding: str = "utf-8",
) -> Path:
    """
    Export selected articles as CSV (flat) or Markdown (grouped by law).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = build_selected_articles_table(df, cfg)

    if fmt == "csv":
        table.to_csv(out_path, index=False, encoding=encoding)
        return out_path

    if fmt == "md":
        # grouped, human-friendly
        lines: list[str] = []
        for law_id, g in table.groupby(cfg.law_id_col, sort=False):
            law_title = g["law_title"].iloc[0] if len(g) else ""
            lines.append(f"# {law_title}".strip())
            lines.append(f"*law_id:* {law_id}")
            lines.append("")

            for _, r in g.iterrows():
                lines.append(f"## {r['article_label']}".strip())
                lines.append("")
                lines.append("**Texte**")
                lines.append("")
                lines.append("```")
                lines.append(str(r["text"] or "").strip())
                lines.append("```")
                lines.append("")
                lines.append("**Ciblage & instruments (article_target_blocks)**")
                lines.append("")
                lines.append("```")
                lines.append(str(r["article_target_blocks"] or "").strip())
                lines.append("```")
                lines.append("")
                lines.append("**Justification**")
                lines.append("")
                lines.append(str(r["article_justification"] or "").strip())
                lines.append("\n---\n")

        out_path.write_text("\n".join(lines), encoding=encoding)
        return out_path

    raise ValueError(f"Unknown fmt: {fmt}")
