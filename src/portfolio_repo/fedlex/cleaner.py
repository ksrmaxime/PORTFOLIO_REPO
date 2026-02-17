from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from lxml import etree


@dataclass(frozen=True)
class CleanConfig:
    keep_titles: bool = True
    drop_notes: bool = True
    drop_editorial_structures: bool = True
    normalize_whitespace: bool = True


_ART_PREFIX_RE = re.compile(r"^\s*Art\.?\s*", re.IGNORECASE)


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _text_content(el: etree._Element) -> str:
    return "".join(el.itertext())


def clean_akoma_ntoso_xml_to_text(xml_path: str | Path, cfg: CleanConfig | None = None) -> str:
    cfg = cfg or CleanConfig()
    xml_path = Path(xml_path)

    raw = xml_path.read_bytes()
    if not raw.strip():
        raise ValueError(f"Empty XML file: {xml_path}")

    head = raw[:2000].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:200]:
        raise ValueError(f"Got HTML instead of XML (likely error page): {xml_path}")

    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        root = etree.fromstring(raw, parser=parser)
    except Exception as e:
        raise ValueError(f"XML parse failed for {xml_path}: {e}") from e

    if root is None:
        raise ValueError(f"Parsed document has no root element: {xml_path}")

    # Drop notes / editorial blocks (structure-based)
    if cfg.drop_notes:
        for n in root.xpath(
            "//*[local-name()='note' or local-name()='authorialNote' or local-name()='editorialNote']"
        ):
            parent = n.getparent()
            if parent is not None:
                parent.remove(n)

    if cfg.drop_editorial_structures:
        for n in root.xpath(
            "//*[local-name()='remark' or local-name()='mod' or local-name()='quotedStructure' or "
            "local-name()='embeddedStructure' or local-name()='commentary']"
        ):
            parent = n.getparent()
            if parent is not None:
                parent.remove(n)

    lines: List[str] = []

    # IMPORTANT: include <level> (fedlex:role="marginal") to keep A./B./I./II. headings.
    containers_xpath = (
        "//*[local-name()='book' or local-name()='title' or local-name()='part' or "
        "local-name()='chapter' or local-name()='subchapter' or local-name()='section' or "
        "local-name()='subsection' or local-name()='division' or local-name()='subdivision' or "
        "local-name()='level' or "
        "local-name()='article']"
    )

    for el in root.xpath(containers_xpath):
        name = el.tag.split("}")[-1]

        num = el.xpath("./*[local-name()='num']")
        heading = el.xpath("./*[local-name()='heading' or local-name()='title']")

        num_s = _text_content(num[0]).strip() if num else ""
        head_s = _text_content(heading[0]).strip() if heading else ""

        # Non-articles: output structural line if we have num/heading
        if name != "article":
            if cfg.keep_titles and (num_s or head_s):
                t = " ".join([p for p in [num_s, head_s] if p]).strip()
                if cfg.normalize_whitespace:
                    t = _norm_ws(t)
                if t:
                    lines.append(t)
            continue

        # Article header
        # Fedlex often already includes "Art. 26" inside <num>, so normalize:
        art_num_clean = _ART_PREFIX_RE.sub("", num_s).strip()
        art_head = head_s.strip()

        art_header = " ".join(
            [p for p in [f"Art. {art_num_clean}".strip() if art_num_clean else "Art.", art_head] if p]
        ).strip()
        if cfg.normalize_whitespace:
            art_header = _norm_ws(art_header)
        if art_header:
            lines.append(art_header)

        # Paragraphs / alinéas
        paras = el.xpath("./*[local-name()='paragraph']")
        for p in paras:
            pnum = p.xpath("./*[local-name()='num']")
            pnum_s = _text_content(pnum[0]).strip() if pnum else ""
            ptext = _text_content(p).strip()
            if cfg.normalize_whitespace:
                ptext = _norm_ws(ptext)
            if not ptext:
                continue
            if pnum_s and not ptext.startswith(pnum_s):
                lines.append(f"{pnum_s} {ptext}")
            else:
                lines.append(ptext)

    return "\n".join(lines).strip() + "\n"
