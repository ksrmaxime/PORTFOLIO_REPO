from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import pandas as pd


# Headings appear at the beginning of a line in clean_text.
_HEADING_MAIN_RE = re.compile(r"^\s*(Section|Chapitre|Titre|Art)\b")


_LEVELS: dict[str, int] = {
    "Partie": 1,
    "Titre": 2,
    "Chapitre": 3,
    "Section": 4,
    "Art": 5,
}


@dataclass(frozen=True)
class Node:
    law_id: str
    node_id: str
    parent_node_id: Optional[str]
    node_type: str  # law_title, partie, titre, chapitre, section, article
    level: int
    order_index: int
    label: str
    text: str
    line_start: int
    line_end: int


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()

_PARTIE_RE = re.compile(
    r"""^\s*(?:
        Partie\b |
        (?:Premi(?:ère|ere)|Deuxi(?:ème|eme)|Troisi(?:ème|eme)|Quatri(?:ème|eme)|Cinqui(?:ème|eme)|
           Sixi(?:ème|eme)|Septi(?:ème|eme)|Huiti(?:ème|eme)|Neuvi(?:ème|eme)|Dixi(?:ème|eme))\s+partie\b |
        (?:\d+)\s*(?:re|e|er|ème|eme)?\s+partie\b |
        (?:[IVXLCDM]+)\s+partie\b
    )\s*:.*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _heading_kind(line: str) -> Optional[str]:
    # Partie d'abord (plus spécifique)
    if _PARTIE_RE.match(line):
        return "Partie"
    m = _HEADING_MAIN_RE.match(line)
    if not m:
        return None
    return m.group(1)


def _node_type_from_heading(heading: str) -> str:
    return {
        "Partie": "partie",
        "Titre": "titre",
        "Chapitre": "chapitre",
        "Section": "section",
        "Art": "article",
    }[heading]


def _split_lines(text: str) -> list[str]:
    # Keep it deterministic; do not strip internal whitespace aggressively here.
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def iter_nodes_for_law(law_id: str, law_title: str, clean_text: str) -> Iterator[Node]:
    """
    Parse a law into hierarchical nodes based on heading lines:
    Partie > Titre > Chapitre > Section > Art

    Assumptions (per user):
    - Headings appear as their own lines (with some phrase on the same line),
      not in the middle of paragraphs.
    """
    order = 0

    # Create a synthetic root node from the law title column.
    root_id = _stable_id(law_id, "law_title", law_title)
    yield Node(
        law_id=law_id,
        node_id=root_id,
        parent_node_id=None,
        node_type="law_title",
        level=0,
        order_index=order,
        label=law_title.strip(),
        text="",
        line_start=0,
        line_end=0,
    )
    order += 1

    stack: list[tuple[int, str]] = [(0, root_id)]  # (level, node_id)
    current_article: Optional[Node] = None
    article_lines: list[str] = []

    lines = _split_lines(clean_text or "")
    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        heading = _heading_kind(line)
        if heading is None:
            # Not a heading: if we are inside an article, accumulate text.
            if current_article is not None:
                article_lines.append(line)
            continue

        # We encountered a heading line. First, flush any open article.
        if current_article is not None:
            flushed = Node(
                **{
                    **current_article.__dict__,
                    "text": "\n".join(article_lines).strip(),
                    "line_end": i - 1,
                }
            )
            yield flushed
            current_article = None
            article_lines = []

        lvl = _LEVELS[heading]
        ntype = _node_type_from_heading(heading)

        # Pop stack to parent level.
        while stack and stack[-1][0] >= lvl:
            stack.pop()

        parent_id = stack[-1][1] if stack else root_id
        label = line.strip()

        node_id = _stable_id(law_id, ntype, str(order), label)

        if ntype == "article":
            # Start a new article; its body is the subsequent non-heading lines.
            current_article = Node(
                law_id=law_id,
                node_id=node_id,
                parent_node_id=parent_id,
                node_type=ntype,
                level=lvl,
                order_index=order,
                label=label,
                text="",  # filled when flushed
                line_start=i,
                line_end=i,  # overwritten when flushed
            )
        else:
            # Structural node: store label, no body text.
            yield Node(
                law_id=law_id,
                node_id=node_id,
                parent_node_id=parent_id,
                node_type=ntype,
                level=lvl,
                order_index=order,
                label=label,
                text="",
                line_start=i,
                line_end=i,
            )
            stack.append((lvl, node_id))

        order += 1

    # Flush last open article if any.
    if current_article is not None:
        yield Node(
            **{
                **current_article.__dict__,
                "text": "\n".join(article_lines).strip(),
                "line_end": len(lines),
            }
        )


def build_structure_parquet_df(
    laws_df: pd.DataFrame,
    law_id_col: str = "base_act_uri",
    title_col: str = "title",
    text_col: str = "clean_text",
) -> pd.DataFrame:
    required = {law_id_col, title_col, text_col}
    missing = required - set(laws_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input parquet: {sorted(missing)}")

    out_rows: list[dict] = []
    for _, row in laws_df.iterrows():
        law_id = str(row[law_id_col])
        law_title = "" if pd.isna(row[title_col]) else str(row[title_col])
        clean_text = "" if pd.isna(row[text_col]) else str(row[text_col])

        for node in iter_nodes_for_law(law_id=law_id, law_title=law_title, clean_text=clean_text):
            out_rows.append(
                {
                    "law_id": node.law_id,
                    "node_id": node.node_id,
                    "parent_node_id": node.parent_node_id,
                    "node_type": node.node_type,
                    "level": node.level,
                    "order_index": node.order_index,
                    "label": node.label,
                    "text": node.text,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                }
            )

    return pd.DataFrame(out_rows)
