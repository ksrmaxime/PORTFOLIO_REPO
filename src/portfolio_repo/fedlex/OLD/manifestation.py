# src/portfolio_repo/fedlex/manifestation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

FEDLEX_SPARQL_ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"

LANG_URI = {
    "de": "http://publications.europa.eu/resource/authority/language/DEU",
    "fr": "http://publications.europa.eu/resource/authority/language/FRA",
    "it": "http://publications.europa.eu/resource/authority/language/ITA",
    "rm": "http://publications.europa.eu/resource/authority/language/ROH",
    "en": "http://publications.europa.eu/resource/authority/language/ENG",
}


def _val(b: Dict[str, Any], key: str) -> Optional[str]:
    v = b.get(key)
    if not isinstance(v, dict):
        return None
    return v.get("value")


def find_xml_manifestations(cc_uri: str, lang: str) -> List[str]:
    """
    Resolve XML (or XML-like) manifestation download URLs for the latest consolidation
    of a Classified Compilation entry (jolux:ConsolidationAbstract).

    IMPORTANT:
    - CC (ConsolidationAbstract) has no Manifestation.
    - Manifestations exist on jolux:Consolidation (work) linked via jolux:isMemberOf.
    """
    lang = lang.lower().strip()
    lang_uri = LANG_URI.get(lang)
    if not lang_uri:
        raise ValueError(f"Unsupported language: {lang} (supported: {sorted(LANG_URI)})")

    sparql = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>

SELECT DISTINCT ?url ?date ?fmt
WHERE {{
  ?work jolux:isMemberOf <{cc_uri}> ;
        jolux:dateApplicability ?date ;
        jolux:isRealizedBy ?expression .

  ?expression jolux:language <{lang_uri}> ;
              jolux:isEmbodiedBy ?manifestation .

  OPTIONAL {{ ?manifestation jolux:format ?fmt . }}
  ?manifestation jolux:isExemplifiedBy ?url .
}}
ORDER BY DESC(?date)
LIMIT 50
"""
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    r = requests.post(
        FEDLEX_SPARQL_ENDPOINT,
        headers=headers,
        data={"query": sparql},
        timeout=60,
    )
    r.raise_for_status()

    bindings = r.json().get("results", {}).get("bindings", [])

    # Collect candidates and filter to XML-ish in Python
    candidates: List[str] = []
    for b in bindings:
        url = _val(b, "url")
        fmt = (_val(b, "fmt") or "").lower()
        if not url:
            continue
        u = url.lower()
        # keep anything that looks like XML (URL or format URI hint)
        if "xml" in u or u.endswith(".xml") or "xml" in fmt or "akn" in fmt or "akoma" in fmt:
            candidates.append(url)

    # If none matched heuristics, still return all URLs (caller can inspect)
    if candidates:
        return candidates

    return [(_val(b, "url")) for b in bindings if _val(b, "url")]
