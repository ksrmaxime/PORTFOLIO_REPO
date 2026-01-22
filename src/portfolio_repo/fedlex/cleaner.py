from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from lxml import etree


@dataclass(frozen=True)
class CleanConfig:
    keep_titles: bool = True
    drop_notes: bool = True
    drop_remarks: bool = True
    normalize_whitespace: bool = True


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

    # Détection très simple des cas où on a reçu du HTML au lieu du XML
    head = raw[:2000].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:200]:
        raise ValueError(f"Got HTML instead of XML (likely error page): {xml_path}")

    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        doc = etree.fromstring(raw, parser=parser)
    except Exception as e:
        raise ValueError(f"XML parse failed for {xml_path}: {e}") from e

    root = doc
    if root is None:
        raise ValueError(f"Parsed document has no root element: {xml_path}")

    # Namespace-agnostic XPath via local-name()
    lines: List[str] = []

    if cfg.drop_notes:
        for n in root.xpath("//*[local-name()='note']"):
            parent = n.getparent()
            if parent is not None:
                parent.remove(n)

    if cfg.drop_remarks:
        for n in root.xpath("//*[local-name()='remark' or local-name()='mod' or local-name()='quotedStructure']"):
            parent = n.getparent()
            if parent is not None:
                parent.remove(n)

    for el in root.xpath(
    "//*[local-name()='book' or local-name()='title' or local-name()='part' or "
    "local-name()='chapter' or local-name()='subchapter' or local-name()='section' or "
    "local-name()='subsection' or local-name()='division' or local-name()='subdivision' or "
    "local-name()='article']"
    ):

        name = el.tag.split("}")[-1]

        num = el.xpath("./*[local-name()='num']")
        heading = el.xpath("./*[local-name()='heading' or local-name()='title']")

        num_s = _text_content(num[0]).strip() if num else ""
        head_s = _text_content(heading[0]).strip() if heading else ""

        if name != "article":
            if cfg.keep_titles and (num_s or head_s):
                t = " ".join([p for p in [num_s, head_s] if p])
                lines.append(t)
            continue

        art_header = " ".join([p for p in [f"Art. {num_s}".strip() if num_s else "Art.", head_s] if p]).strip()
        if art_header:
            lines.append(art_header)

        paras = el.xpath(".//*[local-name()='paragraph']")
        if paras:
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
        else:
            t = _text_content(el).strip()
            if cfg.normalize_whitespace:
                t = _norm_ws(t)
            if t:
                lines.append(t)

        lines.append("")

    out = "\n".join(lines)
    if cfg.normalize_whitespace:
        while "\n\n\n" in out:
            out = out.replace("\n\n\n", "\n\n")
    return out.strip() + "\n"
