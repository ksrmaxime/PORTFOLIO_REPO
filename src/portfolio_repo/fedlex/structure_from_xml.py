from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from lxml import etree


AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
NS = {"akn": AKN_NS}

_ART_PREFIX_RE = re.compile(r"^\s*Art\.?\s*", re.IGNORECASE)


@dataclass(frozen=True)
class Node:
    law_id: str
    node_id: str
    parent_node_id: Optional[str]
    node_type: str
    level: int
    order_index: int
    label: str
    text: str


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def _txt(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _clean_num_art(num_text: str) -> str:
    # Fedlex met souvent "Art. 10" dans <num>. On normalise pour garder seulement le token.
    t = _ART_PREFIX_RE.sub("", (num_text or "")).strip()
    # retire les espaces multiples
    return " ".join(t.split())


def _drop_notes_inplace(root: etree._Element) -> None:
    # enlève authorialNote/editorialNote/note pour ne pas polluer les titres/textes
    for n in root.xpath(
        "//*[local-name()='note' or local-name()='authorialNote' or local-name()='editorialNote']"
    ):
        p = n.getparent()
        if p is not None:
            p.remove(n)


def _article_body_text(article: etree._Element) -> str:
    # On reconstruit un texte d’article raisonnable:
    # - concat paragraphes
    # - si un paragraphe a <num>, on le préfixe
    paras = article.xpath("./akn:paragraph", namespaces=NS)
    out: list[str] = []
    for p in paras:
        pnum = _txt(p.find("./akn:num", namespaces=NS))
        ptxt = " ".join(_txt(p).split())
        if not ptxt:
            continue
        if pnum and not ptxt.startswith(pnum):
            out.append(f"{pnum} {ptxt}")
        else:
            out.append(ptxt)
    return "\n".join(out).strip()


def _level_label(el: etree._Element) -> str:
    num = _txt(el.find("./akn:num", namespaces=NS))
    head = _txt(el.find("./akn:heading", namespaces=NS))
    lab = " ".join([x for x in [num, head] if x]).strip()
    return " ".join(lab.split())


def _struct_label(el: etree._Element) -> str:
    # title/chapter/section etc.
    num = _txt(el.find("./akn:num", namespaces=NS))
    head = _txt(el.find("./akn:heading", namespaces=NS))
    lab = " ".join([x for x in [num, head] if x]).strip()
    return " ".join(lab.split())


def iter_nodes_from_xml(law_id: str, law_title: str, xml_path: Path) -> Iterator[Node]:
    """
    Build structure nodes from Akoma Ntoso XML, preserving the true hierarchy:
    title/chapter/section/level/article.

    Key: article titles often live in <level><heading> that contains <article>.
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    _drop_notes_inplace(root)

    order = 0
    root_id = _stable_id(law_id, "law_title", law_title)

    yield Node(
        law_id=law_id,
        node_id=root_id,
        parent_node_id=None,
        node_type="law_title",
        level=0,
        order_index=order,
        label=(law_title or "").strip(),
        text="",
    )
    order += 1

    # Stack stores (depth_level_int, node_id)
    stack: list[tuple[int, str]] = [(0, root_id)]

    # Mapping tags to (node_type, level_int)
    # We map <level> as node_type="section" to keep compatibility with your downstream schema.
    tag_map: dict[str, tuple[str, int]] = {
        "part": ("partie", 1),
        "title": ("titre", 2),
        "chapter": ("chapitre", 3),
        "section": ("section", 4),
        "subsection": ("section", 4),
        "division": ("section", 4),
        "subdivision": ("section", 4),
        "level": ("section", 5),  # this is the crucial one
        "article": ("article", 6),
    }

    # Traverse in document order within the body
    body = root.find(".//akn:body", namespaces=NS)
    if body is None:
        return

    def walk(el: etree._Element) -> Iterator[etree._Element]:
        for ch in el:
            local = ch.tag.split("}")[-1]
            if local in tag_map:
                yield ch
                yield from walk(ch)

    for el in walk(body):
        local = el.tag.split("}")[-1]
        node_type, lvl = tag_map[local]

        # Choose label & text
        if local == "level":
            label = _level_label(el)
            if not label:
                continue  # avoid empty
            text = ""
        elif local == "article":
            num_el = el.find("./akn:num", namespaces=NS)
            art_token = _clean_num_art(_txt(num_el))
            label = f"Art. {art_token}".strip() if art_token else "Art."
            # IMPORTANT: do NOT include title here; titles come from surrounding <level> or <heading>
            text = _article_body_text(el)
        else:
            label = _struct_label(el)
            if not label:
                continue
            text = ""

        # parent: pop to lvl-1
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        parent_id = stack[-1][1] if stack else root_id

        node_id = _stable_id(law_id, node_type, str(order), label)

        yield Node(
            law_id=law_id,
            node_id=node_id,
            parent_node_id=parent_id,
            node_type=node_type,
            level=lvl,
            order_index=order,
            label=label,
            text=text,
        )
        order += 1

        # push non-articles on stack
        if node_type != "article":
            stack.append((lvl, node_id))


def build_structure_from_catalog(
    catalog_parquet: Path,
    xml_dir: Path,
    out_parquet: Path,
    *,
    law_id_col: str = "base_act_uri",
    title_col: str = "title",
    xml_filename_col: str = "xml_filename",
) -> pd.DataFrame:
    """
    catalog_parquet must provide at least:
      - base_act_uri
      - title
      - xml_filename (or you adapt below)
    xml_dir contains the XML files.
    """
    df = pd.read_parquet(catalog_parquet).copy()

    if law_id_col not in df.columns:
        raise ValueError(f"Missing column {law_id_col} in {catalog_parquet}")
    if title_col not in df.columns:
        df[title_col] = ""
    if xml_filename_col not in df.columns:
        # fallback: derive from consolidation_uri filename if present
        if "consolidation_uri" not in df.columns:
            raise ValueError(
                f"Missing {xml_filename_col} and consolidation_uri in {catalog_parquet}; "
                "need one to locate XML files."
            )
        df[xml_filename_col] = df["consolidation_uri"].astype(str).apply(lambda s: f"{Path(s).name}.xml")

    rows: list[dict] = []

    for _, r in df.iterrows():
        law_id = str(r[law_id_col])
        law_title = "" if pd.isna(r[title_col]) else str(r[title_col])
        xml_file = xml_dir / str(r[xml_filename_col])

        if not xml_file.exists():
            # keep going; you can log missing files separately if needed
            continue

        for node in iter_nodes_from_xml(law_id=law_id, law_title=law_title, xml_path=xml_file):
            rows.append(
                {
                    "law_id": node.law_id,
                    "node_id": node.node_id,
                    "parent_node_id": node.parent_node_id,
                    "node_type": node.node_type,
                    "level": node.level,
                    "order_index": node.order_index,
                    "label": node.label,
                    "text": node.text,
                }
            )

    out = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    return out
