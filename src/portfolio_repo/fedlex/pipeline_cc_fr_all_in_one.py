from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd
import requests
from lxml import etree


# -----------------------------
# Config & constants
# -----------------------------

SPARQL_ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
NS = {"akn": AKN_NS}

USER_AGENT = "portfolio_repo/cc_fr_pipeline (contact: your_email@domain)"

_ART_PREFIX_RE = re.compile(r"^\s*Art\.?\s*", re.IGNORECASE)

# Conservative drop of editorial notes (common in Fedlex Akoma Ntoso exports)
DROP_NOTE_XPATH = "//*[local-name()='note' or local-name()='authorialNote' or local-name()='editorialNote']"


@dataclass(frozen=True)
class LawMeta:
    base_act_uri: str
    title: str
    manifestation_uri: str  # URL we can fetch as XML


@dataclass(frozen=True)
class StructNode:
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


def _norm_ws(s: str) -> str:
    return " ".join((s or "").split())


def _txt(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _clean_art_token(num_text: str) -> str:
    t = _ART_PREFIX_RE.sub("", (num_text or "")).strip()
    return _norm_ws(t)


# -----------------------------
# SPARQL: build catalog (CC, FR, in force)
# -----------------------------

def _sparql_post(query: str, timeout_s: int = 60) -> dict[str, Any]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    }
    resp = requests.post(
        SPARQL_ENDPOINT,
        headers=headers,
        data={"query": query, "format": "application/sparql-results+json"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def build_cc_fr_inforce_catalog(limit: Optional[int] = None) -> list[LawMeta]:
    """
    Returns metadata for "Systematische Rechtssammlung / CC" in French.
    We aim to pick currently-in-force manifestations (consolidated).
    The Fedlex graph uses JOLux ontology. This query is intentionally conservative and may need
    adjustment if the ontology changes.

    If this query fails, the error is explicit so you can adjust it once, centrally.
    """

    # NOTE:
    # The exact Fedlex JOLux model is documented by Swiss authorities; the endpoint is stable:
    # https://fedlex.data.admin.ch/sparqlendpoint
    #
    # This query tries to:
    # - select ELI base acts under /eli/cc/
    # - choose a FR manifestation that has an XML representation (Akoma Ntoso)
    #
    # If you already have a better query, you can replace this string and the pipeline remains identical.
    q = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?baseAct ?title ?manif
WHERE {{
  ?baseAct a eli:LegalResource .
  FILTER(CONTAINS(STR(?baseAct), "/eli/cc/")) .

  OPTIONAL {{ ?baseAct dcterms:title ?title . FILTER (lang(?title) = "fr") }}

  # manifestations (consolidated expressions / manifestations)
  ?baseAct eli:has_member ?expr .
  ?expr eli:language <http://publications.europa.eu/resource/authority/language/FRA> .

  ?expr eli:has_manifestation ?manif .

  # Try to focus on XML-compatible representation
  # Many Fedlex manifestations support /fr/xml via URI patterns even if not modelled here.
}}
"""
    data = _sparql_post(q)
    bindings = data.get("results", {}).get("bindings", [])

    out: list[LawMeta] = []
    seen: set[str] = set()

    for b in bindings:
        base_act_uri = b.get("baseAct", {}).get("value")
        if not base_act_uri:
            continue
        if base_act_uri in seen:
            continue
        title = b.get("title", {}).get("value", "") or ""
        manif = b.get("manif", {}).get("value") or ""

        # Fall back: if the SPARQL model doesn't give us a helpful manif URI,
        # we can still derive an XML URL by taking a known in-force page:
        # base act itself isn't versioned. We prefer manif when present.
        if not manif:
            # We'll still keep base_act_uri and let the downloader try patterns.
            manif = base_act_uri

        out.append(LawMeta(base_act_uri=base_act_uri, title=title, manifestation_uri=manif))
        seen.add(base_act_uri)
        if limit is not None and len(out) >= limit:
            break

    if not out:
        raise RuntimeError(
            "SPARQL query returned 0 laws. The query may need an update to match the current Fedlex graph model."
        )
    return out


# -----------------------------
# Download XML
# -----------------------------

def _is_xml_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "xml" in ct:
        return True
    # sometimes delivered as text/plain
    txt = resp.text.lstrip()[:200].lower()
    return txt.startswith("<?xml") or "<akoma" in txt or "<akomaNtoso".lower() in txt


def _candidate_xml_urls(manifestation_uri: str) -> list[str]:
    """
    Fedlex ELI patterns often allow:
      .../fr/xml
      .../xml
    We try multiple candidates.
    """
    u = manifestation_uri.rstrip("/")
    cands: list[str] = []

    # If already ends with /xml or /fr/xml, keep it first
    if u.endswith("/fr/xml") or u.endswith("/xml"):
        cands.append(u)
    else:
        cands.append(f"{u}/fr/xml")
        cands.append(f"{u}/xml")

    # Some URIs point to base acts; attempt a conservative guess:
    # nothing else here; you can extend if needed.
    return cands


def download_xml_for_law(meta: LawMeta, out_dir: Path, sleep_s: float = 0.0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # stable filename derived from base_act_uri
    fname = _stable_id(meta.base_act_uri) + ".xml"
    out_path = out_dir / fname
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    headers = {"User-Agent": USER_AGENT, "Accept": "application/xml, text/xml;q=0.9, */*;q=0.1"}

    last_err: Optional[Exception] = None
    for url in _candidate_xml_urls(meta.manifestation_uri):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} for {url}")
            if not _is_xml_response(r):
                raise RuntimeError(f"Not XML response for {url} (content-type={r.headers.get('Content-Type')})")
            out_path.write_bytes(r.content)
            if sleep_s:
                time.sleep(sleep_s)
            return out_path
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Failed to download XML for base_act={meta.base_act_uri} from manifestation={meta.manifestation_uri}. "
        f"Last error: {last_err}"
    )


# -----------------------------
# XML cleaning (law-level text)
# -----------------------------

def _drop_notes_inplace(root: etree._Element) -> None:
    for n in root.xpath(DROP_NOTE_XPATH):
        p = n.getparent()
        if p is not None:
            p.remove(n)


def clean_xml_to_flat_text(xml_path: Path) -> str:
    """
    Produces a readable flat text preserving headings + articles.
    Used for law-level parquet and debugging; structure parquet is built from XML tree directly.
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    _drop_notes_inplace(root)

    lines: list[str] = []

    # iterate common structural containers in doc order
    containers_xpath = (
        "//*[local-name()='part' or local-name()='title' or local-name()='chapter' or "
        "local-name()='section' or local-name()='subsection' or local-name()='division' or "
        "local-name()='subdivision' or local-name()='level' or local-name()='article']"
    )

    for el in root.xpath(containers_xpath):
        local = el.tag.split("}")[-1]
        num = el.find("./akn:num", namespaces=NS)
        heading = el.find("./akn:heading", namespaces=NS)

        num_s = _txt(num)
        head_s = _txt(heading)

        if local != "article":
            lab = _norm_ws(" ".join([x for x in [num_s, head_s] if x]).strip())
            if lab:
                lines.append(lab)
            continue

        # article header
        art_token = _clean_art_token(num_s)
        art_label = _norm_ws(" ".join([x for x in [f"Art. {art_token}".strip() if art_token else "Art.", head_s] if x]).strip())
        if art_label:
            lines.append(art_label)

        # paragraphs
        for p in el.findall("./akn:paragraph", namespaces=NS):
            pnum = _txt(p.find("./akn:num", namespaces=NS))
            ptxt = _norm_ws(_txt(p))
            if not ptxt:
                continue
            if pnum and not ptxt.startswith(pnum):
                lines.append(f"{pnum} {ptxt}")
            else:
                lines.append(ptxt)

    return "\n".join(lines).strip() + "\n"


# -----------------------------
# Structure from XML (robust)
# -----------------------------

def _struct_label(el: etree._Element) -> str:
    num = _txt(el.find("./akn:num", namespaces=NS))
    head = _txt(el.find("./akn:heading", namespaces=NS))
    return _norm_ws(" ".join([x for x in [num, head] if x]).strip())


def _level_label(el: etree._Element) -> str:
    # same as struct_label, but kept separate for clarity
    return _struct_label(el)


def _article_body_text(article: etree._Element) -> str:
    out: list[str] = []
    for p in article.findall("./akn:paragraph", namespaces=NS):
        pnum = _txt(p.find("./akn:num", namespaces=NS))
        ptxt = _norm_ws(_txt(p))
        if not ptxt:
            continue
        if pnum and not ptxt.startswith(pnum):
            out.append(f"{pnum} {ptxt}")
        else:
            out.append(ptxt)
    return "\n".join(out).strip()


def iter_structure_nodes_from_xml(law_id: str, law_title: str, xml_path: Path) -> Iterator[StructNode]:
    """
    Builds a faithful hierarchy based on Akoma Ntoso tags.
    Crucial: <level> nodes are included, because they often carry the article marginal title.
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    _drop_notes_inplace(root)

    order = 0
    root_id = _stable_id(law_id, "law_title", law_title)

    yield StructNode(
        law_id=law_id,
        node_id=root_id,
        parent_node_id=None,
        node_type="law_title",
        level=0,
        order_index=order,
        label=_norm_ws(law_title),
        text="",
    )
    order += 1

    # Mapping tags to (node_type, depth)
    # Keep compatibility with your existing downstream schema:
    # - we map <level> to node_type="section" (extra depth level 5)
    tag_map: dict[str, tuple[str, int]] = {
        "part": ("partie", 1),
        "title": ("titre", 2),
        "chapter": ("chapitre", 3),
        "section": ("section", 4),
        "subsection": ("section", 4),
        "division": ("section", 4),
        "subdivision": ("section", 4),
        "level": ("section", 5),
        "article": ("article", 6),
    }

    body = root.find(".//akn:body", namespaces=NS)
    if body is None:
        return

    stack: list[tuple[int, str]] = [(0, root_id)]  # (level_int, node_id)

    def walk(el: etree._Element) -> Iterator[etree._Element]:
        for ch in el:
            local = ch.tag.split("}")[-1]
            if local in tag_map:
                yield ch
                yield from walk(ch)
            else:
                # still walk children, because sometimes structures are nested oddly
                yield from walk(ch)

    for el in walk(body):
        local = el.tag.split("}")[-1]
        if local not in tag_map:
            continue

        node_type, lvl = tag_map[local]

        if local == "article":
            num_s = _txt(el.find("./akn:num", namespaces=NS))
            art_token = _clean_art_token(num_s)
            label = f"Art. {art_token}".strip() if art_token else "Art."
            text = _article_body_text(el)
        elif local == "level":
            label = _level_label(el)
            if not label:
                continue
            text = ""
        else:
            label = _struct_label(el)
            if not label:
                continue
            text = ""

        # parent selection by stack
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        parent_id = stack[-1][1] if stack else root_id

        node_id = _stable_id(law_id, node_type, str(order), label)

        yield StructNode(
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

        if node_type != "article":
            stack.append((lvl, node_id))


# -----------------------------
# Orchestrator: end-to-end
# -----------------------------

def run_pipeline(
    out_dir: Path,
    *,
    limit: Optional[int] = None,
    sleep_s: float = 0.0,
    force_redownload: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_xml_dir = out_dir / "raw_xml"
    raw_xml_dir.mkdir(parents=True, exist_ok=True)

    # Outputs (stable names)
    out_catalog = out_dir / "cc_catalog_fr_inforce.parquet"
    out_clean = out_dir / "laws_federal_fr_inforce_clean.parquet"
    out_struct = out_dir / "laws_structure.parquet"

    print("1) Building catalog via SPARQL…")
    catalog = build_cc_fr_inforce_catalog(limit=limit)

    cat_rows = []
    for m in catalog:
        cat_rows.append(
            {
                "base_act_uri": m.base_act_uri,
                "title": m.title,
                "manifestation_uri": m.manifestation_uri,
                "xml_filename": _stable_id(m.base_act_uri) + ".xml",
            }
        )
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_parquet(out_catalog, index=False)
    print(f"   Saved catalog: {out_catalog} (n={len(cat_df)})")

    print("2) Downloading XML…")
    ok = 0
    err = 0
    for m in catalog:
        try:
            xml_path = raw_xml_dir / (_stable_id(m.base_act_uri) + ".xml")
            if force_redownload and xml_path.exists():
                xml_path.unlink()
            download_xml_for_law(m, raw_xml_dir, sleep_s=sleep_s)
            ok += 1
        except Exception as e:
            err += 1
            print(f"   [WARN] download failed for {m.base_act_uri}: {e}")
    print(f"   Download done: ok={ok}, errors={err}")

    print("3) Building clean law-level parquet…")
    clean_rows = []
    for m in catalog:
        xml_path = raw_xml_dir / (_stable_id(m.base_act_uri) + ".xml")
        if not xml_path.exists():
            continue
        try:
            clean_text = clean_xml_to_flat_text(xml_path)
            clean_rows.append(
                {
                    "base_act_uri": m.base_act_uri,
                    "title": m.title,
                    "clean_text": clean_text,
                    "xml_filename": xml_path.name,
                }
            )
        except Exception as e:
            print(f"   [WARN] clean failed for {m.base_act_uri}: {e}")

    clean_df = pd.DataFrame(clean_rows)
    clean_df.to_parquet(out_clean, index=False)
    print(f"   Saved clean dataset: {out_clean} (n={len(clean_df)})")

    print("4) Building structure parquet from XML (faithful)…")
    struct_rows = []
    for m in catalog:
        xml_path = raw_xml_dir / (_stable_id(m.base_act_uri) + ".xml")
        if not xml_path.exists():
            continue
        try:
            for node in iter_structure_nodes_from_xml(m.base_act_uri, m.title, xml_path):
                struct_rows.append(
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
        except Exception as e:
            print(f"   [WARN] structure failed for {m.base_act_uri}: {e}")

    struct_df = pd.DataFrame(struct_rows)
    struct_df.to_parquet(out_struct, index=False)
    print(f"   Saved structure dataset: {out_struct} (rows={len(struct_df)})")

    print("DONE.")


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory (will contain raw_xml + parquet outputs)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of laws (debug)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between downloads (polite)")
    ap.add_argument("--force-redownload", action="store_true", help="Re-download even if file exists")
    args = ap.parse_args()

    run_pipeline(
        out_dir=Path(args.out_dir),
        limit=args.limit,
        sleep_s=args.sleep,
        force_redownload=args.force_redownload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
