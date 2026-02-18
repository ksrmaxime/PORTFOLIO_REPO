from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import requests
from lxml import etree


# ============================================================
# Helpers
# ============================================================

def _stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _norm_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _local(el: etree._Element) -> str:
    return el.tag.split("}")[-1]


def _txt(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return _norm_ws("".join(el.itertext()))


def _first_xpath_text(root: etree._Element, xpaths: List[str]) -> str:
    for xp in xpaths:
        try:
            res = root.xpath(xp)
        except Exception:
            continue
        if not res:
            continue
        # res can be list of elements or strings/attrs
        if isinstance(res[0], etree._Element):
            t = _txt(res[0])
        else:
            t = _norm_ws(str(res[0]))
        if t:
            return t
    return ""


def _drop_notes_inplace(root: etree._Element) -> None:
    for n in root.xpath("//*[local-name()='note' or local-name()='authorialNote' or local-name()='editorialNote']"):
        p = n.getparent()
        if p is not None:
            p.remove(n)


def _looks_like_html(content: bytes) -> bool:
    head = content[:800].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:200]


def _looks_like_xml(content: bytes) -> bool:
    head = content.lstrip()[:200]
    return head.startswith(b"<") and not _looks_like_html(content)


_ART_PREFIX_RE = None


def _clean_art_token(num_text: str) -> str:
    # "Art. 86a" -> "86a"
    t = (num_text or "").strip()
    t = t.replace("\u00a0", " ")
    t = t.strip()
    if t.lower().startswith("art"):
        # remove leading "Art", "Art."
        t = t.split(None, 1)[-1] if len(t.split(None, 1)) == 2 else ""
        t = t.lstrip(".").strip()
    return _norm_ws(t)


# ============================================================
# SPARQL (ancienne logique)
# ============================================================

@dataclass(frozen=True)
class SparqlConfig:
    endpoint: str = "https://fedlex.data.admin.ch/sparqlendpoint"
    timeout_s: int = 60
    max_retries: int = 4
    backoff_s: float = 1.5


class SparqlClient:
    def __init__(self, cfg: SparqlConfig | None = None) -> None:
        self.cfg = cfg or SparqlConfig()

    def query_json(self, sparql: str) -> Dict[str, Any]:
        headers = {"Accept": "application/sparql-results+json"}
        data = {"query": sparql}

        last_err: Optional[Exception] = None
        for i in range(self.cfg.max_retries):
            try:
                r = requests.post(self.cfg.endpoint, data=data, headers=headers, timeout=self.cfg.timeout_s)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff_s * (i + 1))
        raise RuntimeError(f"SPARQL query failed after retries: {last_err}") from last_err


def _extract_last_segment(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def build_cc_catalog_fr(client: SparqlClient) -> pd.DataFrame:
    """
    Catalogue CC, FR, lois fédérales, en vigueur.
    -> 1 ligne par base_act_uri (dernière consolidation).
    """
    sparql = """
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>

SELECT ?base ?cons ?typeDoc ?status ?title_fr ?title_any
WHERE {
  ?cons jolux:isMemberOf ?base .
  FILTER(STRSTARTS(STR(?base), "https://fedlex.data.admin.ch/eli/cc/"))
  FILTER(REGEX(STR(?cons), "/[0-9]{8}$"))

  # Lois fédérales uniquement
  ?base jolux:typeDocument ?typeDoc .
  VALUES ?typeDoc {
    <https://fedlex.data.admin.ch/vocabulary/resource-type/21>
    <https://fedlex.data.admin.ch/vocabulary/resource-type/22>
  }

  # En vigueur
  ?base jolux:inForceStatus ?status .
  VALUES ?status {
    <https://fedlex.data.admin.ch/vocabulary/enforcement-status/0>
  }

  OPTIONAL {
    ?base jolux:title ?title_fr .
    FILTER(lang(?title_fr) = "fr")
  }
  OPTIONAL { ?base jolux:title ?title_any . }
}
"""
    data = client.query_json(sparql)
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        raise RuntimeError("SPARQL returned 0 rows for CC catalog (laws in force).")

    records: List[dict] = []
    for b in bindings:
        base = b.get("base", {}).get("value")
        cons = b.get("cons", {}).get("value")
        if base is None or cons is None:
            continue
        records.append(
            {
                "base_act_uri": base,
                "consolidation_uri": cons,
                "type_doc": b.get("typeDoc", {}).get("value"),
                "in_force_status_uri": b.get("status", {}).get("value"),
                "title_fr": b.get("title_fr", {}).get("value"),
                "title_any": b.get("title_any", {}).get("value"),
            }
        )

    df = pd.DataFrame.from_records(records)
    df["consolidation_date_yyyymmdd"] = df["consolidation_uri"].map(_extract_last_segment).astype(str)

    # dernière consolidation
    df = (
        df.sort_values(["base_act_uri", "consolidation_date_yyyymmdd"])
        .groupby("base_act_uri", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # titre (fallback)
    df["title"] = df["title_fr"].fillna(df["title_any"])
    df["lang"] = "fr"
    return df


# ============================================================
# Download XML (ancienne logique jolux:isExemplifiedBy)
# ============================================================

@dataclass(frozen=True)
class DownloadConfig:
    timeout_s: int = 60
    user_agent: str = "portfolio_repo/1.0 (fedlex downloader)"
    overwrite: bool = False
    cache_manifestation_resolution: bool = True


def _manifestation_uri(consolidation_uri: str, lang: str = "fr") -> str:
    return f"{consolidation_uri.rstrip('/')}/{lang}/xml"


def _resolve_filestore_url(
    sparql: SparqlClient,
    manifestation: str,
    cache: Optional[Dict[str, Optional[str]]] = None,
) -> Optional[str]:
    if cache is not None and manifestation in cache:
        return cache[manifestation]

    q = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>

SELECT ?file
WHERE {{
  <{manifestation}> jolux:isExemplifiedBy ?file .
}}
LIMIT 1
"""
    js = sparql.query_json(q)
    bindings = js.get("results", {}).get("bindings", [])
    if not bindings:
        if cache is not None:
            cache[manifestation] = None
        return None
    file_url = bindings[0].get("file", {}).get("value")
    if cache is not None:
        cache[manifestation] = file_url
    return file_url


def download_cc_xml_batch(
    catalog: pd.DataFrame,
    out_raw_dir: Path,
    cfg: DownloadConfig,
) -> pd.DataFrame:
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
    }

    sparql = SparqlClient()
    cache: Optional[Dict[str, Optional[str]]] = {} if cfg.cache_manifestation_resolution else None

    rows: List[Dict[str, Any]] = []

    for _, row in catalog.iterrows():
        base_act_uri = str(row["base_act_uri"])
        cons_uri = str(row["consolidation_uri"])
        cons_date = str(row["consolidation_date_yyyymmdd"])
        ident = base_act_uri.rstrip("/").split("/")[-1]
        out_path = out_raw_dir / f"{cons_date}__{ident}.xml"

        if out_path.exists() and not cfg.overwrite:
            rows.append(
                dict(
                    base_act_uri=base_act_uri,
                    cons_date=cons_date,
                    manifestation_uri=_manifestation_uri(cons_uri, "fr"),
                    file_url=None,
                    ok=True,
                    skipped_existing=True,
                    path=str(out_path),
                    sha256=None,
                    http_status=None,
                    error=None,
                )
            )
            continue

        manifestation = _manifestation_uri(cons_uri, "fr")
        try:
            file_url = _resolve_filestore_url(sparql, manifestation, cache=cache)
            if not file_url:
                rows.append(
                    dict(
                        base_act_uri=base_act_uri,
                        cons_date=cons_date,
                        manifestation_uri=manifestation,
                        file_url=None,
                        ok=False,
                        skipped_existing=False,
                        path=None,
                        sha256=None,
                        http_status=None,
                        error="No jolux:isExemplifiedBy file URL for manifestation",
                    )
                )
                continue

            r = requests.get(file_url, headers=headers, timeout=cfg.timeout_s, allow_redirects=True)
            content = r.content
            ct = (r.headers.get("Content-Type") or "").lower()

            if r.status_code != 200:
                rows.append(
                    dict(
                        base_act_uri=base_act_uri,
                        cons_date=cons_date,
                        manifestation_uri=manifestation,
                        file_url=file_url,
                        ok=False,
                        skipped_existing=False,
                        path=None,
                        sha256=None,
                        http_status=r.status_code,
                        error=f"HTTP {r.status_code}",
                    )
                )
                continue

            if not _looks_like_xml(content):
                rows.append(
                    dict(
                        base_act_uri=base_act_uri,
                        cons_date=cons_date,
                        manifestation_uri=manifestation,
                        file_url=file_url,
                        ok=False,
                        skipped_existing=False,
                        path=None,
                        sha256=None,
                        http_status=r.status_code,
                        error=f"Non-XML content (ct={ct}, html={_looks_like_html(content)})",
                    )
                )
                continue

            out_path.write_bytes(content)
            rows.append(
                dict(
                    base_act_uri=base_act_uri,
                    cons_date=cons_date,
                    manifestation_uri=manifestation,
                    file_url=file_url,
                    ok=True,
                    skipped_existing=False,
                    path=str(out_path),
                    sha256=_sha256_bytes(content),
                    http_status=r.status_code,
                    error=None,
                )
            )

        except Exception as e:
            rows.append(
                dict(
                    base_act_uri=base_act_uri,
                    cons_date=cons_date,
                    manifestation_uri=manifestation,
                    file_url=None,
                    ok=False,
                    skipped_existing=False,
                    path=None,
                    sha256=None,
                    http_status=None,
                    error=str(e),
                )
            )

    return pd.DataFrame(rows)


# ============================================================
# Title fallback from XML (fix law_title label vide)
# ============================================================

def extract_law_title_from_xml(xml_path: Path) -> str:
    raw = xml_path.read_bytes()
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(raw, parser=parser)
    _drop_notes_inplace(root)

    # Try common AkomaNtoso metadata places
    # (Fedlex can vary; we keep it robust with multiple fallbacks)
    candidates = [
        "//*[local-name()='docTitle'][1]",
        "//*[local-name()='meta']//*[local-name()='identification']//*[local-name()='FRBRWork']//*[local-name()='FRBRname']/@value",
        "//*[local-name()='meta']//*[local-name()='identification']//*[local-name()='FRBRWork']//*[local-name()='FRBRalias']/@value",
        "//*[local-name()='meta']//*[local-name()='identification']//*[local-name()='FRBRWork']//*[local-name()='FRBRthis']/@value",
        # As last resort: first big heading in body
        "//*[local-name()='body']//*[local-name()='heading'][1]",
    ]
    title = _first_xpath_text(root, candidates)
    return title


# ============================================================
# Structure from XML (namespace-agnostic) + article text extraction
# ============================================================

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


def _struct_label(el: etree._Element) -> str:
    num_el = el.xpath("./*[local-name()='num'][1]")
    head_el = el.xpath("./*[local-name()='heading'][1]")
    num = _txt(num_el[0]) if num_el else ""
    head = _txt(head_el[0]) if head_el else ""
    return _norm_ws(" ".join([x for x in [num, head] if x]))


def _article_label(article: etree._Element) -> str:
    num_el = article.xpath("./*[local-name()='num'][1]")
    num = _txt(num_el[0]) if num_el else ""
    token = _clean_art_token(num)
    return f"Art. {token}".strip() if token else "Art."


def _article_body_text(article: etree._Element) -> str:
    # Robust: paragraphs might contain <content>, lists, etc.
    # We collect paragraph nodes and format with their <num> when present.
    out: List[str] = []
    paras = article.xpath(".//*[local-name()='paragraph']")
    for p in paras:
        pnum_el = p.xpath("./*[local-name()='num'][1]")
        pnum = _txt(pnum_el[0]) if pnum_el else ""
        ptxt = _txt(p)
        if not ptxt:
            continue
        # avoid duplicating num if itertext already includes it
        if pnum and ptxt.startswith(pnum):
            out.append(ptxt)
        elif pnum:
            out.append(f"{pnum} {ptxt}")
        else:
            out.append(ptxt)
    return "\n".join(out).strip()


def iter_structure_nodes_from_xml(law_id: str, law_title: str, xml_path: Path) -> Iterator[StructNode]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    _drop_notes_inplace(root)

    # find body namespace-agnostically
    body_list = root.xpath("//*[local-name()='body'][1]")
    if not body_list:
        return
    body = body_list[0]

    # Tag mapping (keep your schema)
    tag_map: Dict[str, Tuple[str, int]] = {
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

    stack: List[Tuple[int, str]] = [(0, root_id)]

    def walk(el: etree._Element) -> Iterator[etree._Element]:
        for ch in el:
            local = _local(ch)
            if local in tag_map:
                yield ch
                yield from walk(ch)
            else:
                # still walk inside (Fedlex sometimes nests structures under wrappers)
                yield from walk(ch)

    for el in walk(body):
        local = _local(el)
        if local not in tag_map:
            continue

        node_type, lvl = tag_map[local]

        if local == "article":
            label = _article_label(el)
            text = _article_body_text(el)
        else:
            label = _struct_label(el)
            if not label:
                continue
            text = ""

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


# ============================================================
# Orchestration end-to-end + final flat export
# ============================================================

def run_all(
    out_dir: Path,
    *,
    overwrite_download: bool = False,
    limit: Optional[int] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_xml"
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_catalog = out_dir / "cc_catalog_fr_inforce.parquet"
    out_download_log = out_dir / "download_log.parquet"
    out_struct_raw = out_dir / "laws_structure_raw.parquet"

    out_final = out_dir / "laws_structure_final.parquet"
    out_final_csv = out_dir / "laws_structure_final.csv"

    print("1) Building catalog via SPARQL (known-working query)…")
    sp = SparqlClient()
    catalog = build_cc_catalog_fr(sp)
    if limit is not None:
        catalog = catalog.head(limit).copy()
    catalog.to_parquet(out_catalog, index=False)
    print(f"   Saved: {out_catalog} (n={len(catalog)})")

    print("2) Downloading XML via manifestation -> jolux:isExemplifiedBy…")
    dl_cfg = DownloadConfig(overwrite=overwrite_download)
    dl_log = download_cc_xml_batch(catalog, raw_dir, dl_cfg)
    dl_log.to_parquet(out_download_log, index=False)
    ok = int(dl_log["ok"].sum()) if "ok" in dl_log.columns else 0
    print(f"   Saved: {out_download_log} (ok={ok}, total={len(dl_log)})")

    # map law_id -> xml path
    path_map = (
        dl_log.loc[dl_log["ok"] == True, ["base_act_uri", "path"]]  # noqa: E712
        .dropna()
        .set_index("base_act_uri")["path"]
        .to_dict()
    )

    print("3) Building structure dataset (XML-native, includes <level> titles)…")
    struct_rows: List[dict] = []
    missing_title = 0

    for _, row in catalog.iterrows():
        law_id = str(row["base_act_uri"])
        title = "" if pd.isna(row.get("title")) else str(row.get("title") or "")
        xml_path_s = path_map.get(law_id)
        if not xml_path_s:
            continue
        xml_path = Path(xml_path_s)

        # Title fallback from XML if catalog title missing
        if not _norm_ws(title):
            title = extract_law_title_from_xml(xml_path)
            if not title:
                missing_title += 1
                title = ""  # keep empty if impossible

        try:
            for n in iter_structure_nodes_from_xml(law_id, title, xml_path):
                struct_rows.append(
                    dict(
                        law_id=n.law_id,
                        node_id=n.node_id,
                        parent_node_id=n.parent_node_id,
                        node_type=n.node_type,
                        level=n.level,
                        order_index=n.order_index,
                        label=n.label,
                        text=n.text,
                    )
                )
        except Exception as e:
            print(f"   [WARN] structure failed for {law_id}: {e}")

    struct_df = pd.DataFrame(struct_rows)
    struct_df.to_parquet(out_struct_raw, index=False)
    print(f"   Saved: {out_struct_raw} (rows={len(struct_df)})")
    if missing_title:
        print(f"   [INFO] Titles missing even after XML fallback: {missing_title}")

    print("4) Building FINAL flat file (with level 0 + article texts)…")
    final_df = struct_df[["law_id", "node_type", "level", "order_index", "label", "text"]].copy()

    # ensure non-articles have empty text
    final_df.loc[final_df["node_type"] != "article", "text"] = ""

    # stable order per law
    final_df.sort_values(["law_id", "order_index"], inplace=True)
    final_df["order_index"] = final_df.groupby("law_id").cumcount()

    final_df.to_parquet(out_final, index=False)
    final_df.to_csv(out_final_csv, index=False, encoding="utf-8-sig")

    print(f"   Saved: {out_final}")
    print(f"   Saved: {out_final_csv}")
    print("DONE.")


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--overwrite-download", action="store_true", help="Re-download XML even if already present")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of laws (debug)")
    args = ap.parse_args()

    run_all(
        out_dir=Path(args.out_dir),
        overwrite_download=args.overwrite_download,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
