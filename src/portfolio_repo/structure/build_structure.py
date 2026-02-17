from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd


# Headings appear at the beginning of a line in clean_text.
_HEADING_MAIN_RE = re.compile(r"^\s*(Section|Chapitre|Titre|Art)\b", re.IGNORECASE)

# Sub-headings / titres marginaux:
# - "A. ...", "B. ..."
# - "Cbis. ...", "Dter. ..."
# - "I. ...", "II. ..."
# - "1. ...", "2. ..."
_HEADING_MARGINAL_RE = re.compile(
    r"^\s*(?:"
    r"\d+"
    r"|[IVXLCDM]{1,10}"
    r"|[A-Z](?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)?"
    r")\.\s+\S",
    re.IGNORECASE,
)

# Parse "Art. 26 <optional title>"
# group1 = article number token (e.g., "26", "4a", "86a", "86f")
# group2 = optional title (rest of line)
_ART_LINE_RE = re.compile(r"^\s*Art\.?\s+(\S+)(?:\s+(.*\S))?\s*$", re.IGNORECASE)

_LEVELS: dict[str, int] = {
    "Partie": 1,
    "Titre": 2,
    "Chapitre": 3,
    "Section": 4,
    "Marginal": 5,  # inserted layer (A./I./1./Article titles)
    "Art": 6,
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
    s = line.strip()

    if _PARTIE_RE.match(s):
        return "Partie"

    if _HEADING_MARGINAL_RE.match(s):
        return "Marginal"

    m = _HEADING_MAIN_RE.match(s)
    if not m:
        return None
    return m.group(1).capitalize()


def _node_type_from_heading(heading: str) -> str:
    # Keep schema stable: Marginal is stored as node_type="section"
    return {
        "Partie": "partie",
        "Titre": "titre",
        "Chapitre": "chapitre",
        "Section": "section",
        "Marginal": "section",
        "Art": "article",
    }[heading]


def _split_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def iter_nodes_for_law(law_id: str, law_title: str, clean_text: str) -> Iterator[Node]:
    """
    Parse a law into hierarchical nodes based on heading lines:
    Partie > Titre > Chapitre > Section > Marginal > Art

    Plus: si une ligne d'article est de la forme "Art. X <titre>", on la découpe en:
      - Marginal: "<titre>"
      - Article: "Art. X"
    afin d'avoir une structure homogène (titre sur sa ligne / Art sur sa ligne).
    """
    order = 0

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

    stack: list[tuple[int, str]] = [(0, root_id)]
    current_article: Optional[Node] = None
    article_lines: list[str] = []

    lines = _split_lines(clean_text or "")
    for i, raw in enumerate(lines, start=1):
        line_raw = raw.rstrip("\n")
        line = line_raw.strip()

        if not line:
            if current_article is not None:
                article_lines.append("")
            continue

        heading = _heading_kind(line_raw)
        if heading is None:
            if current_article is not None:
                article_lines.append(line)
            continue

        # Flush open article
        if current_article is not None:
            yield Node(
                **{
                    **current_article.__dict__,
                    "text": "\n".join(article_lines).strip(),
                    "line_end": i - 1,
                }
            )
            current_article = None
            article_lines = []

        lvl = _LEVELS[heading]

        # Compute parent: pop stack to the correct level
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        parent_id = stack[-1][1] if stack else root_id

        # Special handling for "Art. X <title>"
        if heading == "Art":
            m = _ART_LINE_RE.match(line_raw)
            art_token = m.group(1).strip() if m else ""
            art_title = (m.group(2).strip() if (m and m.group(2)) else "")

            # Normalize the article label to "Art. <token>" only
            art_label = f"Art. {art_token}".strip() if art_token else line

            # If there is an inline title, create a Marginal node for it first
            # (so title appears on its own line in structure), then attach the article under it.
            if art_title:
                marginal_lvl = _LEVELS["Marginal"]  # 5
                marginal_type = _node_type_from_heading("Marginal")  # "section"

                # Pop stack to marginal parent (between Section and Art)
                while stack and stack[-1][0] >= marginal_lvl:
                    stack.pop()
                marginal_parent = stack[-1][1] if stack else root_id

                marginal_node_id = _stable_id(law_id, marginal_type, str(order), art_title)
                yield Node(
                    law_id=law_id,
                    node_id=marginal_node_id,
                    parent_node_id=marginal_parent,
                    node_type=marginal_type,
                    level=marginal_lvl,
                    order_index=order,
                    label=art_title,
                    text="",
                    line_start=i,
                    line_end=i,
                )
                stack.append((marginal_lvl, marginal_node_id))
                order += 1

                # Article parent becomes the marginal title we just pushed
                parent_id = marginal_node_id

            # Create the article node (body collected after this line)
            node_id = _stable_id(law_id, "article", str(order), art_label)
            current_article = Node(
                law_id=law_id,
                node_id=node_id,
                parent_node_id=parent_id,
                node_type="article",
                level=_LEVELS["Art"],
                order_index=order,
                label=art_label,
                text="",
                line_start=i,
                line_end=i,
            )
            order += 1
            continue

        # Default handling: structural node (Partie/Titre/Chapitre/Section/Marginal)
        ntype = _node_type_from_heading(heading)
        label = line
        node_id = _stable_id(law_id, ntype, str(order), label)

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

    # Flush last open article
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
