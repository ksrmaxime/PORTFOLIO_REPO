from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Accept the common variants:
# - "Art. 1" / "Art.1"
# - "Art 1"
# - "Article 1"
# - "ARTICLE 1"
#
# Also accept suffixes: a, bis, ter, quater, etc.
_ART_RE = re.compile(
    r"(?mi)^\s*(?:Art\.?|Article)\s*(\d+)\s*([a-z]{1,6})?\b"
)

# Some Fedlex conversions may inline headings; a fallback that finds "Art." not necessarily at line start.
_ART_FALLBACK_RE = re.compile(
    r"(?mi)(?:^|[\n\r ])(?:Art\.?|Article)\s*(\d+)\s*([a-z]{1,6})?\b"
)


@dataclass(frozen=True)
class ArticleBlock:
    art_id: str
    art_number: int
    art_suffix: str
    text: str
    start_char: int
    end_char: int


def _normalize_art_id(num: str, suffix: str | None) -> tuple[str, int, str]:
    n = int(num)
    s = (suffix or "").strip().lower()
    # normalize frequent forms: "a", "bis", "ter", etc.
    art_id = f"{n}{s}" if s else f"{n}"
    return art_id, n, s


def split_by_article(text: str) -> tuple[Optional[str], list[ArticleBlock]]:
    matches = list(_ART_RE.finditer(text))
    if not matches:
        # try fallback if headings are not clean line-starts
        matches = list(_ART_FALLBACK_RE.finditer(text))
        if not matches:
            return (text.strip() if text.strip() else None), []

    preamble = text[: matches[0].start()].strip()
    preamble_text = preamble if preamble else None

    blocks: list[ArticleBlock] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        num = m.group(1)
        suf = m.group(2) or ""
        art_id, art_number, art_suffix = _normalize_art_id(num, suf)

        art_text = text[start:end].strip()
        blocks.append(
            ArticleBlock(
                art_id=art_id,
                art_number=art_number,
                art_suffix=art_suffix,
                text=art_text,
                start_char=start,
                end_char=end,
            )
        )

    return preamble_text, blocks

