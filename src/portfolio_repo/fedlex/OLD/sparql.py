from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datetime import date

import requests

FEDLEX_SPARQL_ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"
VOCAB_LEGAL_RESOURCE_GENRE = "https://fedlex.data.admin.ch/vocabulary/legal-resource-genre"

# Known core concept observed in the Fedlex vocabulary UI:
# "Federal law" -> .../legal-resource-genre/100
FALLBACK_FEDERAL_LAW_GENRE = "https://fedlex.data.admin.ch/vocabulary/legal-resource-genre/100"


@dataclass(frozen=True)
class SparqlClient:
    endpoint: str = FEDLEX_SPARQL_ENDPOINT
    timeout_s: int = 90

    def query(self, sparql: str) -> Dict[str, Any]:
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        r = requests.post(self.endpoint, headers=headers, data={"query": sparql}, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()


def _val(binding: Dict[str, Any], key: str) -> Optional[str]:
    v = binding.get(key)
    if not isinstance(v, dict):
        return None
    return v.get("value")


def find_federal_act_genre_terms() -> List[str]:
    """
    Return URI(s) of Act-Type concept(s) representing federal acts/laws
    from the Act Types vocabulary (legal-resource-genre).
    """
    sparql = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?term ?label
WHERE {{
  ?term a skos:Concept ;
        skos:inScheme <{VOCAB_LEGAL_RESOURCE_GENRE}> ;
        skos:prefLabel ?label .

  FILTER(
    CONTAINS(LCASE(STR(?label)), "federal law")
    || CONTAINS(LCASE(STR(?label)), "bundesgesetz")
    || CONTAINS(LCASE(STR(?label)), "loi féd")
    || CONTAINS(LCASE(STR(?label)), "loi fed")
    || CONTAINS(LCASE(STR(?label)), "federal act")
  )
}}
"""
    client = SparqlClient()
    out = client.query(sparql)
    bindings = out.get("results", {}).get("bindings", [])
    terms = [_val(b, "term") for b in bindings]
    terms = [t for t in terms if t]

    # Robust fallback
    if not terms:
        terms = [FALLBACK_FEDERAL_LAW_GENRE]

    return terms

# ... keep SparqlClient and _val as you already have ...

FEDERAL_ACT_TYPES = [
    "https://fedlex.data.admin.ch/vocabulary/resource-type/21",  # Federal act / Loi fédérale
    "https://fedlex.data.admin.ch/vocabulary/resource-type/22",  # Emergency federal act / Loi fédérale urgente
    "https://fedlex.data.admin.ch/vocabulary/resource-type/35",  # Emergency federal act subject to mandatory referendum
]


def list_federal_acts(limit: Optional[int] = None) -> List[Dict[str, Optional[str]]]:
    lim = f"LIMIT {int(limit)}" if limit else ""
    today = date.today().isoformat()
    values = " ".join(f"<{t}>" for t in FEDERAL_ACT_TYPES)

    sparql = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>


SELECT DISTINCT ?cc ?type ?dateEntry ?title ?titleLang
WHERE {{
  ?cc a jolux:ConsolidationAbstract ;
      jolux:typeDocument ?type ;
      jolux:dateEntryInForce ?dateEntry .

  VALUES ?type {{ {values} }}

  FILTER(?dateEntry <= "{today}"^^xsd:date)

    OPTIONAL {{
    ?cc jolux:isRealizedBy ?expr .
    # Prefer French label/title
    OPTIONAL {{
      ?expr dcterms:title ?t1 .
      FILTER(LANG(?t1) = "fr")
    }}
    OPTIONAL {{
      ?expr <http://www.w3.org/2000/01/rdf-schema#label> ?t2 .
      FILTER(LANG(?t2) = "fr")
    }}
    BIND(COALESCE(?t1, ?t2) AS ?title)
    BIND("fr" AS ?titleLang)
  }}
}}
{lim}
"""
    client = SparqlClient()
    out = client.query(sparql)
    bindings = out.get("results", {}).get("bindings", [])

    rows: List[Dict[str, Optional[str]]] = []
    seen = set()
    for b in bindings:
        cc = _val(b, "cc")
        if not cc or cc in seen:
            continue
        seen.add(cc)
        rows.append(
            {
                "eli_uri": cc,
                "type_uri": _val(b, "type"),
                "date_entry_in_force": _val(b, "dateEntry"),
                "title": _val(b, "title"),
                "title_lang": _val(b, "titleLang"),
            }
        )
    return rows

