from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import hashlib
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ExtractConfig:
    """
    Extract only normative structure + content:
    - Exclude preface/preamble entirely
    - Exclude authorialNote (and a few other editorial tags) during rendering
    - Keep chapter/section/article hierarchy headings
    """
    skip_tags: Tuple[str, ...] = (
        "authorialNote",
        "mod",          # sometimes used for amendments markup
        "meta",         # just in case a subtree is walked by mistake
    )
    include_heading_nodes: Tuple[str, ...] = (
        "book", "part", "title", "chapter", "section", "subsection",
    )
    include_article_node: str = "article"
    paragraph_nodes: Tuple[str, ...] = ("paragraph", "subparagraph", "point", "indent")
    # Formatting
    blank_line_between_blocks: bool = True


AKN_NS = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _local(tag: str) -> str:
    # '{ns}article' -> 'article'
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _itertext_skip(elem: ET.Element, skip_locals: set[str]) -> Iterator[str]:
    """
    Like itertext(), but skips subtrees whose local tag is in skip_locals.
    """
    if _local(elem.tag) in skip_locals:
        return
    if elem.text:
        yield elem.text
    for child in list(elem):
        yield from _itertext_skip(child, skip_locals)
        if child.tail:
            yield child.tail


def _clean_ws(s: str) -> str:
    # conservative whitespace normalization
    s = s.replace("\u00a0", " ")  # non-breaking space
    lines = [ln.strip() for ln in s.splitlines()]
    # remove empty lines at ends, keep internal empties only if meaningful
    out_lines: List[str] = []
    for ln in lines:
        if ln == "":
            # collapse multiple blank lines
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
        else:
            out_lines.append(" ".join(ln.split()))
    # trim trailing blank
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    while out_lines and out_lines[0] == "":
        out_lines.pop(0)
    return "\n".join(out_lines).strip()


def _text_of(node: Optional[ET.Element], skip_locals: set[str]) -> str:
    if node is None:
        return ""
    return _clean_ws("".join(_itertext_skip(node, skip_locals)))


def _find_one(root: ET.Element, xpath: str) -> Optional[ET.Element]:
    return root.find(xpath, AKN_NS)


def _find_all(root: ET.Element, xpath: str) -> List[ET.Element]:
    return root.findall(xpath, AKN_NS)


def parse_akn_xml(path: str | Path) -> ET.Element:
    path = Path(path)
    tree = ET.parse(path)
    return tree.getroot()


def extract_law_identity(root: ET.Element) -> Dict[str, str]:
    """
    Best-effort identity fields from AKN metadata.
    You can override law_id upstream if you already have it from registry.
    """
    # FRBR URI (this is usually stable)
    frbr_this = _find_one(root, ".//akn:FRBRWork/akn:FRBRthis")
    eli_uri = frbr_this.get("value", "") if frbr_this is not None else ""

    # language (Akoma Ntoso often has xml:lang on akomaNtoso or act)
    lang = root.get("{http://www.w3.org/XML/1998/namespace}lang", "") or ""

    # Title: try docTitle first, then shortTitle
    doc_title = _find_one(root, ".//akn:identification/akn:FRBRExpression/akn:FRBRname")
    title = doc_title.get("value", "") if doc_title is not None else ""

    if not title:
        doc_title2 = _find_one(root, ".//akn:docTitle")
        title = _text_of(doc_title2, set()) if doc_title2 is not None else ""

    return {
        "eli_uri": eli_uri,
        "lang": lang,
        "title": title,
    }


def _render_heading_block(node: ET.Element, skip_locals: set[str]) -> str:
    """
    Render chapter/section headings as:
    <num> <heading>
    """
    num = _text_of(_find_one(node, "akn:num"), skip_locals)
    heading = _text_of(_find_one(node, "akn:heading"), skip_locals)

    if num and heading:
        return f"{num}  {heading}"
    if num:
        return num
    if heading:
        return heading
    return ""


def _render_article(article: ET.Element, cfg: ExtractConfig) -> Dict[str, Any]:
    skip_locals = set(cfg.skip_tags)

    eid = article.get("eId", "") or ""

    art_num = _text_of(_find_one(article, "akn:num"), skip_locals)
    art_heading = _text_of(_find_one(article, "akn:heading"), skip_locals)

    header = ""
    if art_num and art_heading:
        header = f"{art_num}  {art_heading}"
    elif art_num:
        header = art_num
    elif art_heading:
        header = art_heading

    # paragraphs can be nested and varied; do a robust ordered walk:
    blocks: List[str] = []
    if header:
        blocks.append(header)

    # Common AKN pattern: article/content/p, and/or article/paragraph/content/p
    # We'll walk through known paragraph-like nodes in document order.
    def render_content_text(container: ET.Element) -> str:
        # Prefer content node if present
        content = _find_one(container, "akn:content")
        if content is not None:
            return _text_of(content, skip_locals)
        return _text_of(container, skip_locals)

    para_like = []
    for tag in cfg.paragraph_nodes:
        para_like.extend(article.findall(f".//akn:{tag}", AKN_NS))

    # If none found, fallback to direct content
    if not para_like:
        direct = _find_one(article, "akn:content")
        t = _text_of(direct, skip_locals) if direct is not None else ""
        if t:
            blocks.append(t)
    else:
        # Render each para-like node as "<num> <text>" where possible.
        for pnode in para_like:
            # Only render "top-level-ish" ones: avoid rendering nested ones twice
            # Simple heuristic: skip if parent is also paragraph-like
            parent = _get_parent(article, pnode)
            if parent is not None and _local(parent.tag) in set(cfg.paragraph_nodes):
                continue

            pnum = _text_of(_find_one(pnode, "akn:num"), skip_locals)
            ptext = render_content_text(pnode)

            if not ptext:
                continue

            if pnum:
                blocks.append(f"{pnum} {ptext}")
            else:
                blocks.append(ptext)

    clean_text_article = _clean_ws("\n".join(blocks))
    return {
        "article_eid": eid,
        "article_num": art_num,
        "article_heading": art_heading,
        "clean_text_article": clean_text_article,
    }


def _get_parent(root: ET.Element, child: ET.Element) -> Optional[ET.Element]:
    # ElementTree has no parent pointer; small helper
    for node in root.iter():
        for c in list(node):
            if c is child:
                return node
    return None


def extract_normative_structure(
    xml_path: str | Path,
    cfg: ExtractConfig = ExtractConfig(),
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns:
      law_record: dict with clean_text (full law, headings + articles)
      article_records: list of dicts (one per article)
    """
    xml_path = Path(xml_path)
    root = parse_akn_xml(xml_path)

    ident = extract_law_identity(root)

    # Only take act/body (exclude preface/preamble by not traversing them)
    body = _find_one(root, ".//akn:act/akn:body")
    if body is None:
        raise ValueError(f"No act/body found in {xml_path.name} (unexpected AKN structure).")

    skip_locals = set(cfg.skip_tags)

    # Traverse body in order, capturing headings (chapter/section/...) and articles.
    law_blocks: List[str] = []
    articles: List[Dict[str, Any]] = []

    def walk(node: ET.Element, hierarchy: List[str]) -> None:
        for child in list(node):
            lt = _local(child.tag)

            if lt in cfg.include_heading_nodes:
                h = _render_heading_block(child, skip_locals)
                new_hierarchy = hierarchy[:]
                if h:
                    law_blocks.append(h)
                    new_hierarchy.append(h)
                    if cfg.blank_line_between_blocks:
                        law_blocks.append("")
                walk(child, new_hierarchy)
                continue

            if lt == cfg.include_article_node:
                art = _render_article(child, cfg)
                art["hierarchy_path"] = " > ".join(hierarchy)
                articles.append(art)

                if art["clean_text_article"]:
                    law_blocks.append(art["clean_text_article"])
                    if cfg.blank_line_between_blocks:
                        law_blocks.append("")
                continue

            # otherwise just walk through (containers like
