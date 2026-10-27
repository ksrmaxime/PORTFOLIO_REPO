from __future__ import annotations

import argparse
import gc
import hashlib
import re
import resource
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
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


def _peak_rss_mb() -> float:
    # ru_maxrss : Ko sous Linux, octets sous macOS
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


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
            # garder le texte qui suit la note (lxml le stocke dans n.tail)
            if n.tail:
                prev = n.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or "") + n.tail
                else:
                    p.text = (p.text or "") + n.tail
            p.remove(n)
    # éléments de liste (<item>: "a.", "b.", tirets) : marqueur de saut de ligne,
    # converti en "\n" dans _article_body_text (sinon "règle:a. la mise…;b. l'utilisation…")
    for it in root.xpath("//*[local-name()='item']"):
        it.text = _ITEM_MARK + (it.text or "")
    # <br/> -> espace (sinon "Loi fédérale<br/>sur" devient "Loi fédéralesur")
    for br in root.xpath("//*[local-name()='br']"):
        br.tail = " " + (br.tail or "")


def _looks_like_html(content: bytes) -> bool:
    head = content[:800].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:200]


def _looks_like_xml(content: bytes) -> bool:
    head = content.lstrip()[:200]
    return head.startswith(b"<") and not _looks_like_html(content)


_ART_PREFIX_RE = None

# label de la ligne unique créée pour un acte sans <article> (cf. iter_structure_nodes_from_xml)
BODY_FALLBACK_LABEL = "Texte intégral (sans articles)"

# caractère à usage privé (non blanc, donc préservé par _norm_ws)
_ITEM_MARK = "\ue000"


def _restore_items(s: str) -> str:
    s = s.replace(" " + _ITEM_MARK, _ITEM_MARK).replace(_ITEM_MARK + " ", _ITEM_MARK)
    return s.replace(_ITEM_MARK, "\n").strip()


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


# Types de documents Fedlex (vocabulaire jolux:typeDocument, resource-type/<code>).
# code -> (catégorie, libellé officiel FR)
RESOURCE_TYPE_BASE = "https://fedlex.data.admin.ch/vocabulary/resource-type/"

DOC_TYPES: "OrderedDict[int, Tuple[str, str]]" = OrderedDict(
    [
        # Droit interne
        (21, ("loi", "Loi fédérale")),
        (22, ("loi", "Loi fédérale urgente")),
        (28, ("ordonnance", "Ordonnance de l'Assemblée fédérale")),
        (29, ("ordonnance", "Ordonnance du Conseil fédéral")),
        (26, ("ordonnance", "Ordonnance d'un département")),
        (27, ("ordonnance", "Ordonnance d'un office")),
        # Actes (ordonnances, règlements) d'autres autorités fédérales
        (1, ("autre_acte", "Acte législatif d'exploitations ou d'établissements autonomes")),
        (2, ("autre_acte", "Acte législatif de Commissions")),
        (3, ("autre_acte", "Acte législatif des Tribunaux fédéraux")),
        (4, ("autre_acte", "Acte législatif du Conseil des États")),
        (5, ("autre_acte", "Acte législatif du Conseil national")),
        (15, ("convention_cantons", "Conventions entre la Confédération et les cantons")),
        # Droit international (RS 0.xxx)
        (31, ("traite_international", "Traité international bilatéral")),
        (32, ("traite_international", "Traité international multilatéral")),
    ]
)

# Arrêtés fédéraux : publiés dans la Feuille fédérale (FF, eli/fga), jamais dans le RS.
# Pas de version consolidée ni de statut « en vigueur » : on prend tous ceux dont la date de
# l'acte (jolux:dateDocument) est comprise entre FGA_MIN_DATE et as_of ET qui sont publiés en
# XML (la FF n'existe en XML que depuis mi-2021 ; les PDF antérieurs sont écartés).
FGA_DOC_TYPES: "OrderedDict[int, Tuple[str, str]]" = OrderedDict(
    [
        (6, ("arrete_federal", "Arrêté du Conseil fédéral")),
        (7, ("arrete_federal", "Arrêté fédéral simple (approbation de traités)")),
        (8, ("arrete_federal", "Arrêté fédéral simple (autres)")),
        (9, ("arrete_federal", "Arrêté fédéral sujet au référendum facultatif (autres)")),
        (10, ("arrete_federal", "Arrêté fédéral soumis au référendum obligatoire")),
        (36, ("arrete_federal", "Arrêté fédéral sujet au référendum facultatif (traités)")),
        (62, ("arrete_federal", "Arrêté fédéral simple")),
        (65, ("arrete_federal", "Arrêté fédéral urgent sans base constitutionnelle, durée >1 an")),
        (66, ("arrete_federal", "Arrêté fédéral urgent sans base constitutionnelle, durée <1 an")),
        (74, ("arrete_federal", "Arrêté fédéral urgent avec base constitutionnelle, durée >1 an")),
        (75, ("arrete_federal", "Arrêté fédéral urgent avec base constitutionnelle, durée <1 an")),
        (76, ("arrete_federal", "Arrêté fédéral de portée générale")),
    ]
)
FGA_MIN_DATE = "2015-01-01"

ALL_DOC_TYPES: "OrderedDict[int, Tuple[str, str]]" = OrderedDict(list(DOC_TYPES.items()) + list(FGA_DOC_TYPES.items()))

_INTERNATIONAL = {"traite_international"}

DOC_TYPE_GROUPS: Dict[str, List[int]] = {
    "lois": [c for c, (cat, _) in DOC_TYPES.items() if cat == "loi"],
    "ordonnances": [c for c, (cat, _) in DOC_TYPES.items() if cat == "ordonnance"],
    "autres_actes": [c for c, (cat, _) in DOC_TYPES.items() if cat == "autre_acte"],
    "conventions_cantons": [c for c, (cat, _) in DOC_TYPES.items() if cat == "convention_cantons"],
    "traites": [c for c, (cat, _) in DOC_TYPES.items() if cat in _INTERNATIONAL],
    "arretes": list(FGA_DOC_TYPES.keys()),
    # tout le droit interne : RS interne + arrêtés fédéraux (FF)
    "interne": [c for c, (cat, _) in DOC_TYPES.items() if cat not in _INTERNATIONAL] + list(FGA_DOC_TYPES.keys()),
    # défaut : RS (droit interne + traités) + arrêtés fédéraux (FF)
    "all": list(DOC_TYPES.keys()) + list(FGA_DOC_TYPES.keys()),
}

# Fedlex ne renseigne pas jolux:inForceStatus pour beaucoup d'actes récents (ex. ONum,
# RS 172.019.1, en vigueur depuis le 1.5.2025). Sans statut, un acte est considéré en
# vigueur s'il n'a pas de date de fin (dateNoLongerInForce) et que son entrée en vigueur
# est comprise entre cette date et as_of. Le plancher écarte ~900 vieux actes (1940-1990)
# abrogés mais jamais mis à jour dans les métadonnées (arrêtés de guerre, etc.).
NO_STATUS_MIN_ENTRY = "2000-01-01"


def build_cc_catalog_fr(
    client: SparqlClient,
    type_codes: Optional[List[int]] = None,
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """
    Catalogue du Recueil systématique (RS, eli/cc), FR, actes en vigueur, des types
    demandés (par défaut : tout, droit interne + traités, cf. DOC_TYPES / DOC_TYPE_GROUPS).
    En vigueur = statut Fedlex 'en vigueur', ou, faute de statut, cf. NO_STATUS_MIN_ENTRY.
    -> 1 ligne par base_act_uri : la dernière consolidation dont la date est <= as_of
       (YYYYMMDD, défaut = aujourd'hui). Les consolidations datées dans le futur
       (versions déjà publiées mais pas encore en vigueur) sont ignorées.
    """
    type_codes = list(type_codes or DOC_TYPES.keys())
    unknown = [c for c in type_codes if c not in DOC_TYPES]
    if unknown:
        raise ValueError(f"Unknown resource-type codes: {unknown} (known: {list(DOC_TYPES)})")
    as_of = as_of or date.today().strftime("%Y%m%d")

    as_of_iso = f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
    values = "\n    ".join(f"<{RESOURCE_TYPE_BASE}{c}>" for c in type_codes)
    sparql = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>

SELECT ?base ?cons ?typeDoc ?status ?entry ?end ?title_fr
WHERE {{
  ?base jolux:typeDocument ?typeDoc .
  VALUES ?typeDoc {{
    {values}
  }}
  # Recueil systématique (RS) uniquement
  FILTER(STRSTARTS(STR(?base), "https://fedlex.data.admin.ch/eli/cc/"))

  # Vigueur : statut 'en vigueur' OU (pas de statut ET entrée en vigueur récente) ;
  # dans les deux cas, pas de date de fin <= as_of
  OPTIONAL {{ ?base jolux:inForceStatus ?status . }}
  OPTIONAL {{ ?base jolux:dateEntryInForce ?entry . }}
  OPTIONAL {{ ?base jolux:dateNoLongerInForce ?end . }}
  BIND(COALESCE(STR(?status), "") AS ?st)
  BIND(COALESCE(STR(?entry), "") AS ?en)
  BIND(COALESCE(STR(?end), "9999-12-31") AS ?ed)
  FILTER(?ed > "{as_of_iso}")
  FILTER(?st = "https://fedlex.data.admin.ch/vocabulary/enforcement-status/0"
         || (?st = "" && ?en != "" && ?en <= "{as_of_iso}" && ?en >= "{NO_STATUS_MIN_ENTRY}"))

  # Versions consolidées
  ?cons jolux:isMemberOf ?base .
  FILTER(REGEX(STR(?cons), "/[0-9]{{8}}$"))

  # Titre FR (porté par l'expression /fr de l'acte)
  OPTIONAL {{
    ?base jolux:isRealizedBy ?expr .
    FILTER(STRENDS(STR(?expr), "/fr"))
    ?expr jolux:title ?title_fr .
  }}
}}
"""
    data = client.query_json(sparql)
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        raise RuntimeError("SPARQL returned 0 rows for CC catalog (acts in force).")

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
                "date_entry_in_force": b.get("entry", {}).get("value"),
                "date_no_longer_in_force": b.get("end", {}).get("value"),
                "title_fr": b.get("title_fr", {}).get("value"),
            }
        )

    df = pd.DataFrame.from_records(records)
    df["consolidation_date_yyyymmdd"] = df["consolidation_uri"].map(_extract_last_segment).astype(str)

    n_acts_all = df["base_act_uri"].nunique()

    # ignorer les consolidations futures (pas encore en vigueur)
    df = df[df["consolidation_date_yyyymmdd"] <= as_of]

    # dernière consolidation <= as_of
    df = (
        df.sort_values(["base_act_uri", "consolidation_date_yyyymmdd"])
        .groupby("base_act_uri", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    n_dropped = n_acts_all - len(df)
    if n_dropped:
        print(f"   [INFO] {n_dropped} acte(s) sans consolidation <= {as_of} ignoré(s)")

    # type de document
    df["type_code"] = df["type_doc"].map(_extract_last_segment).astype(int)
    df["doc_category"] = df["type_code"].map(lambda c: DOC_TYPES[c][0])
    df["doc_type_label"] = df["type_code"].map(lambda c: DOC_TYPES[c][1])

    # base de la décision "en vigueur" (statut Fedlex, ou déduite faute de statut)
    df["in_force_basis"] = df["in_force_status_uri"].map(lambda v: "status" if isinstance(v, str) and v else "inferred_no_status")

    # titres Fedlex parfois balisés ("1<sup>er</sup> mars")
    df["title_fr"] = df["title_fr"].map(lambda t: _norm_ws(re.sub(r"<[^>]+>", "", t)) if isinstance(t, str) else t)
    df["title"] = df["title_fr"]
    df["lang"] = "fr"
    df["as_of"] = as_of
    return df


def build_fga_catalog_fr(
    client: SparqlClient,
    type_codes: List[int],
    as_of: Optional[str] = None,
    date_from: str = FGA_MIN_DATE,
) -> pd.DataFrame:
    """
    Catalogue des arrêtés fédéraux de la Feuille fédérale (eli/fga), FR, dont la date de l'acte
    (jolux:dateDocument) est comprise entre date_from et as_of. Mêmes colonnes que le catalogue
    du RS : l'URI de l'acte sert aussi de « consolidation » (pas de version consolidée en FF).
    """
    unknown = [c for c in type_codes if c not in FGA_DOC_TYPES]
    if unknown:
        raise ValueError(f"Unknown FGA resource-type codes: {unknown}")
    as_of = as_of or date.today().strftime("%Y%m%d")
    as_of_iso = f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
    values = "\n    ".join(f"<{RESOURCE_TYPE_BASE}{c}>" for c in type_codes)
    sparql = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?w ?typeDoc ?d ?title_fr
WHERE {{
  ?w jolux:typeDocument ?typeDoc ;
     jolux:legalResourceFamilyType <https://fedlex.data.admin.ch/vocabulary/resource-family/fga> ;
     jolux:dateDocument ?d .
  VALUES ?typeDoc {{
    {values}
  }}
  FILTER(STR(?d) >= "{date_from}" && STR(?d) <= "{as_of_iso}")
  # seulement les arrêtés publiés en XML (FF au format Akoma Ntoso, depuis mi-2021) :
  # avant, la FF n'existe qu'en PDF / DOC, dont la découpe en articles n'est pas fiable
  ?w jolux:isRealizedBy ?exml .
  FILTER(STRENDS(STR(?exml), "/fr"))
  ?exml jolux:isEmbodiedBy ?mxml .
  FILTER(STRENDS(STR(?mxml), "/fr/xml"))
  ?mxml jolux:isExemplifiedBy ?fxml .
  OPTIONAL {{
    ?w jolux:isRealizedBy ?expr .
    FILTER(STRENDS(STR(?expr), "/fr"))
    ?expr jolux:title ?title_fr .
  }}
}}
"""
    data = client.query_json(sparql)
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        print("   [WARN] SPARQL : 0 arrêté fédéral trouvé dans la Feuille fédérale")
        return pd.DataFrame()
    records = []
    for b in bindings:
        w = b.get("w", {}).get("value")
        d = b.get("d", {}).get("value", "")[:10]
        if not w or not d:
            continue
        records.append(dict(
            base_act_uri=w,
            consolidation_uri=w,
            type_doc=b.get("typeDoc", {}).get("value"),
            in_force_status_uri=None,
            date_entry_in_force=d,
            date_no_longer_in_force=None,
            title_fr=b.get("title_fr", {}).get("value"),
            consolidation_date_yyyymmdd=d.replace("-", ""),
        ))
    df = pd.DataFrame.from_records(records).drop_duplicates("base_act_uri").reset_index(drop=True)
    df["type_code"] = df["type_doc"].map(_extract_last_segment).astype(int)
    df["doc_category"] = df["type_code"].map(lambda c: FGA_DOC_TYPES[c][0])
    df["doc_type_label"] = df["type_code"].map(lambda c: FGA_DOC_TYPES[c][1])
    df["in_force_basis"] = f"fga_date_document_{date_from}_{as_of_iso}"
    df["title_fr"] = df["title_fr"].map(lambda t: _norm_ws(re.sub(r"<[^>]+>", "", t)) if isinstance(t, str) else t)
    df["title"] = df["title_fr"]
    df["lang"] = "fr"
    df["as_of"] = as_of
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
        is_fga = "/eli/fga/" in base_act_uri
        if is_fga:
            ident = "fga_" + base_act_uri.rstrip("/").split("/eli/fga/")[-1].replace("/", "_")
        else:
            # identifiant complet après /eli/cc/ ("2018/385" -> "2018_385") : le seul dernier
            # segment ("385") n'est pas unique (2007/385 = LIB, 2018/385 = ordonnance DFAE) et
            # faisait partager le même fichier XML à deux actes différents.
            ident = base_act_uri.rstrip("/").split("/eli/cc/")[-1].replace("/", "_")

        # format XML (Akoma Ntoso) uniquement, RS comme Feuille fédérale
        candidates = [("xml", ".xml", _looks_like_xml)]

        existing = [out_raw_dir / f"{cons_date}__{ident}{ext}" for _, ext, _ in candidates]
        existing = [p for p in existing if p.exists()]
        if existing and not cfg.overwrite:
            rows.append(dict(base_act_uri=base_act_uri, cons_date=cons_date,
                             manifestation_uri=None, file_url=None, ok=True, skipped_existing=True,
                             path=str(existing[0]), sha256=None, http_status=None, error=None))
            continue

        errors = []
        done = False
        for fmt, ext, check in candidates:
            manifestation = f"{cons_uri.rstrip('/')}/fr/{fmt}"
            out_path = out_raw_dir / f"{cons_date}__{ident}{ext}"
            try:
                file_url = _resolve_filestore_url(sparql, manifestation, cache=cache)
                if not file_url:
                    errors.append(f"{fmt}: pas de fichier (jolux:isExemplifiedBy)")
                    continue
                r = requests.get(file_url, headers=headers, timeout=cfg.timeout_s, allow_redirects=True)
                content = r.content
                if r.status_code != 200:
                    errors.append(f"{fmt}: HTTP {r.status_code}")
                    continue
                if not check(content):
                    ct = (r.headers.get("Content-Type") or "").lower()
                    errors.append(f"{fmt}: contenu inattendu (ct={ct})")
                    continue
                out_path.write_bytes(content)
                rows.append(dict(base_act_uri=base_act_uri, cons_date=cons_date,
                                 manifestation_uri=manifestation, file_url=file_url, ok=True,
                                 skipped_existing=False, path=str(out_path),
                                 sha256=_sha256_bytes(content), http_status=r.status_code, error=None))
                done = True
                break
            except Exception as e:
                errors.append(f"{fmt}: {e}")
        if not done:
            rows.append(dict(base_act_uri=base_act_uri, cons_date=cons_date,
                             manifestation_uri=None, file_url=None, ok=False, skipped_existing=False,
                             path=None, sha256=None, http_status=None, error=" | ".join(errors)))

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


_DOCNUMBER_RE = re.compile(rb"<(?:\w+:)?docNumber[^>]*>(.*?)</(?:\w+:)?docNumber>", re.DOTALL)


def extract_sr_number(xml_path: Path) -> str:
    """Numéro RS (ex. '172.019.1', '0.103.2') depuis <docNumber> du préambule.
    Lecture du seul début du fichier (pas de second parsing XML complet)."""
    with open(xml_path, "rb") as fh:
        head = fh.read(300_000)
    m = _DOCNUMBER_RE.search(head)
    if not m:
        return ""
    txt = re.sub(rb"<[^>]+>", b"", m.group(1)).decode("utf-8", errors="ignore")
    return _norm_ws(txt)


_FRBR_NUMBER_FR_RE = re.compile(rb'<(?:\w+:)?FRBRnumber[^>]*xml:lang="fr"[^>]*value="([^"]+)"')


def extract_ff_number(path: Path, base_act_uri: str) -> str:
    """Référence « FF année numéro » d'un arrêté de la Feuille fédérale (tient lieu de n° RS)."""
    if path.suffix == ".xml":
        with open(path, "rb") as fh:
            m = _FRBR_NUMBER_FR_RE.search(fh.read(100_000))
        if m:
            return _norm_ws(m.group(1).decode("utf-8", errors="ignore"))
    parts = base_act_uri.rstrip("/").split("/eli/fga/")[-1].split("/")
    return f"FF {parts[0]} {parts[1]}" if len(parts) >= 2 else ""


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
    return _norm_ws(" ".join([x for x in [num, head] if x]).replace(_ITEM_MARK, " "))


def _article_label(article: etree._Element) -> str:
    # label = "Art. 27" (sans le heading)
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
        ptxt = _restore_items(_txt(p))
        pnum = _restore_items(pnum)
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
    """
    Structure fidèle XML + amélioration:
    - Si un <article> a un <heading>, on crée un noeud "section" niveau 5 juste avant l'article,
      label = heading, comme un titre marginal (synthetic level).
    - Puis l'article sort avec label = "Art. X" uniquement.
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    _drop_notes_inplace(root)

    body_list = root.xpath("//*[local-name()='body'][1]")
    if not body_list:
        return
    body = body_list[0]

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
    n_articles = 0
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
                yield from walk(ch)

    for el in walk(body):
        local = _local(el)
        if local not in tag_map:
            continue

        node_type, lvl = tag_map[local]

        # ------------ NON-ARTICLE NODES ------------
        if local != "article":
            label = _struct_label(el)
            if not label:
                continue
            text = ""

            # find parent from stack
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

            # push to stack (non-article)
            stack.append((lvl, node_id))
            continue

        # ------------ ARTICLE NODES ------------
        # parent for article is computed from lvl=6
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        parent_id = stack[-1][1] if stack else root_id

        # article heading (marginal title) -> create synthetic section (lvl 5) just before article
        heading_el = el.xpath("./*[local-name()='heading'][1]")
        heading = _txt(heading_el[0]) if heading_el else ""
        heading = _norm_ws(heading)

        synthetic_id: Optional[str] = None

        if heading:
            # Synthetic node behaves like a level 5 title right before the article
            synth_lvl = 5
            synth_type = "section"
            synth_label = heading

            # parent for lvl=5 is the nearest stack level <5
            # (use a temp computation so we don't destroy stack for the article)
            tmp_stack = stack.copy()
            while tmp_stack and tmp_stack[-1][0] >= synth_lvl:
                tmp_stack.pop()
            synth_parent_id = tmp_stack[-1][1] if tmp_stack else root_id

            synthetic_id = _stable_id(law_id, "synthetic_article_heading", str(order), synth_label)
            yield StructNode(
                law_id=law_id,
                node_id=synthetic_id,
                parent_node_id=synth_parent_id,
                node_type=synth_type,
                level=synth_lvl,
                order_index=order,
                label=synth_label,
                text="",
            )
            order += 1

        # Now the real article row
        art_label = _article_label(el)
        art_text = _article_body_text(el)

        art_parent = synthetic_id if synthetic_id else parent_id

        art_id = _stable_id(law_id, "article", str(order), art_label)
        yield StructNode(
            law_id=law_id,
            node_id=art_id,
            parent_node_id=art_parent,
            node_type="article",
            level=6,
            order_index=order,
            label=art_label,
            text=art_text,
        )
        order += 1
        n_articles += 1

    # Repli : acte sans <article> mais avec du texte (protocoles, déclarations,
    # mémorandums, listes techniques, vieux arrêtés numérotés "1.", "2."...).
    # Le texte entier du <body> devient une seule ligne de niveau 6, pour ne pas le perdre.
    if n_articles == 0:
        # un saut de ligne avant chaque bloc numéroté / paragraphe, un espace après chaque numéro
        for el in body.xpath(".//*[local-name()='level' or local-name()='paragraph' or local-name()='hcontainer']"):
            el.text = _ITEM_MARK + (el.text or "")
        for num in body.xpath(".//*[local-name()='num']"):
            num.tail = " " + (num.tail or "")
        body_text = _restore_items(_txt(body))
        if body_text:
            yield StructNode(
                law_id=law_id,
                node_id=_stable_id(law_id, "article", str(order), "body_fallback"),
                parent_node_id=root_id,
                node_type="article",
                level=6,
                order_index=order,
                label=BODY_FALLBACK_LABEL,
                text=body_text,
            )


# ============================================================
# Orchestration end-to-end + final flat export
# ============================================================

def run_all(
    out_dir: Path,
    *,
    overwrite_download: bool = False,
    limit: Optional[int] = None,
    doc_types: str = "all",
    as_of: Optional[str] = None,
    max_mem_gb: Optional[float] = None,
    verbose: bool = False,
) -> None:
    # Plafond mémoire du processus : un acte qui fait exploser la mémoire lève alors
    # MemoryError (acte ignoré et signalé) au lieu de faire tuer tout le job par SLURM.
    if max_mem_gb:
        lim = int(max_mem_gb * 1024**3)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            print(f"   [INFO] plafond mémoire du processus : {max_mem_gb:g} Go")
        except (ValueError, OSError) as e:
            print(f"   [WARN] impossible de fixer le plafond mémoire : {e}")
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
    codes = DOC_TYPE_GROUPS[doc_types]
    cc_codes = [c for c in codes if c in DOC_TYPES]
    fga_codes = [c for c in codes if c in FGA_DOC_TYPES]
    parts = []
    if cc_codes:
        parts.append(build_cc_catalog_fr(sp, type_codes=cc_codes, as_of=as_of))
    if fga_codes:
        fga = build_fga_catalog_fr(sp, type_codes=fga_codes, as_of=as_of)
        print(f"   Arrêtés fédéraux (Feuille fédérale, publiés en XML, depuis {FGA_MIN_DATE}) : {len(fga)}")
        if len(fga):
            parts.append(fga)
    catalog = pd.concat(parts, ignore_index=True)
    print("   " + ", ".join(f"{k}={v}" for k, v in catalog["doc_type_label"].value_counts().items()))
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

    # garde-fou : chaque acte doit avoir son propre fichier XML
    ok_paths = dl_log.loc[dl_log["ok"] == True, ["base_act_uri", "path"]].dropna()  # noqa: E712
    dup = ok_paths[ok_paths["path"].duplicated(keep=False)]
    if len(dup):
        raise RuntimeError(f"Plusieurs actes partagent le même fichier XML :\n{dup.head(10).to_string()}")

    # map law_id -> xml path
    path_map = (
        dl_log.loc[dl_log["ok"] == True, ["base_act_uri", "path"]]  # noqa: E712
        .dropna()
        .set_index("base_act_uri")["path"]
        .to_dict()
    )

    print("3) Building structure dataset (XML-native, includes <level> titles)…")
    # stockage en colonnes (listes) plutôt qu'en liste de dicts : ~3x moins de mémoire
    cols = ["law_id", "node_id", "parent_node_id", "node_type", "level", "order_index", "label", "text"]
    struct_cols: Dict[str, list] = {c: [] for c in cols}
    missing_title = 0
    sr_numbers: Dict[str, str] = {}
    skipped_memory: List[str] = []
    n_catalog = len(catalog)

    for i_act, (_, row) in enumerate(catalog.iterrows(), start=1):
        if i_act % 250 == 0 or i_act == n_catalog:
            gc.collect()
            print(f"   … {i_act:,}/{n_catalog:,} actes | {len(struct_cols['law_id']):,} lignes | "
                  f"mémoire max {_peak_rss_mb():,.0f} Mo", flush=True)
        law_id = str(row["base_act_uri"])
        title = "" if pd.isna(row.get("title")) else str(row.get("title") or "")
        xml_path_s = path_map.get(law_id)
        if not xml_path_s:
            continue
        xml_path = Path(xml_path_s)
        try:
            if "/eli/fga/" in law_id:
                sr_numbers[law_id] = extract_ff_number(xml_path, law_id)
            else:
                sr_numbers[law_id] = extract_sr_number(xml_path)
        except Exception:
            sr_numbers[law_id] = ""

        # Title fallback from XML if catalog title missing
        if not _norm_ws(title):
            title = extract_law_title_from_xml(xml_path)
            if not title:
                missing_title += 1
                title = ""  # keep empty if impossible

        if verbose:
            print(f"   [{i_act}] {law_id} ({xml_path.stat().st_size/1e6:.1f} Mo)", flush=True)
        try:
            nodes = list(iter_structure_nodes_from_xml(law_id, title, xml_path))
        except MemoryError:
            nodes = None
            gc.collect()
            print(f"   [WARN] MÉMOIRE dépassée pour {law_id} ({xml_path}) — acte ignoré", flush=True)
            skipped_memory.append(law_id)
            continue
        except Exception as e:
            print(f"   [WARN] structure failed for {law_id}: {e}")
            continue
        n_chars = sum(len(n.text) + len(n.label) for n in nodes)
        if len(nodes) > 20_000 or n_chars > 20_000_000:
            print(f"   [INFO] acte volumineux {law_id}: {len(nodes):,} lignes, {n_chars/1e6:,.1f} M caractères "
                  f"(XML {xml_path.stat().st_size/1e6:,.1f} Mo)", flush=True)
        for n in nodes:
            for c in cols:
                struct_cols[c].append(getattr(n, c))
        del nodes

    if skipped_memory:
        print(f"   [WARN] {len(skipped_memory)} acte(s) ignoré(s) faute de mémoire : {skipped_memory}", flush=True)
    struct_df = pd.DataFrame(struct_cols)
    del struct_cols
    gc.collect()
    doc_info = catalog.set_index("base_act_uri")[["doc_category", "type_code", "doc_type_label"]]
    struct_df = struct_df.join(doc_info, on="law_id")
    struct_df["sr_number"] = struct_df["law_id"].map(sr_numbers).fillna("")
    struct_df.to_parquet(out_struct_raw, index=False)
    print(f"   Saved: {out_struct_raw} (rows={len(struct_df)}) | mémoire max {_peak_rss_mb():,.0f} Mo", flush=True)
    if missing_title:
        print(f"   [INFO] Titles missing even after XML fallback: {missing_title}")

    print("4) Building FINAL flat file (with level 0 + article texts)…")
    # NB: law_id = URI ELI de l'acte (loi OU ordonnance) ; doc_category distingue les deux.
    final_df = struct_df[
        ["law_id", "sr_number", "doc_category", "type_code", "doc_type_label",
         "node_type", "level", "order_index", "label", "text"]
    ].copy()
    del struct_df
    gc.collect()

    # ensure non-articles have empty text
    # (déjà vide en sortie de iter_structure_nodes_from_xml ; garde-fou sans .loc[...] = "" :
    #  avec les chaînes Arrow de pandas récent, cette affectation convertit toute la colonne
    #  en tableau numpy à largeur fixe -> 133 Go pour 250k lignes dont une de 140k caractères)
    is_article = (final_df["node_type"] == "article").to_numpy()
    final_df["text"] = [t if keep else "" for t, keep in zip(final_df["text"].tolist(), is_article)]

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
    ap.add_argument("--limit", type=int, default=None, help="Limit number of acts (debug)")
    ap.add_argument("--max-mem-gb", type=float, default=None,
                    help="Plafond mémoire du processus (Go) ; un acte qui le dépasse est ignoré et signalé")
    ap.add_argument("--verbose", action="store_true", help="Afficher chaque acte avant sa découpe")
    ap.add_argument(
        "--doc-types",
        choices=sorted(DOC_TYPE_GROUPS),
        default="all",
        help="Types d'actes : all (défaut : RS — droit interne + traités — et arrêtés fédéraux de la FF), "
        "interne (lois, ordonnances, autres actes fédéraux, conventions Confédération-cantons, arrêtés fédéraux), "
        "traites, arretes, ou un groupe : lois, ordonnances, autres_actes, conventions_cantons",
    )
    ap.add_argument(
        "--as-of",
        default=None,
        help="Date de référence YYYYMMDD : version consolidée en vigueur à cette date (défaut : aujourd'hui)",
    )
    args = ap.parse_args()

    run_all(
        out_dir=Path(args.out_dir),
        overwrite_download=args.overwrite_download,
        limit=args.limit,
        doc_types=args.doc_types,
        as_of=args.as_of,
        max_mem_gb=args.max_mem_gb,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
