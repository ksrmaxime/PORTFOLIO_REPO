from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from portfolio_repo.fedlex.sparql import SparqlClient


_BASE_ACT_RE = re.compile(r"^https://fedlex\.data\.admin\.ch/eli/cc/\d{2,4}/[^/]+$")


def _extract_last_segment(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def build_cc_catalog_fr(client: SparqlClient) -> pd.DataFrame:
    sparql = """
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>

SELECT ?base ?cons ?title ?typeDoc ?status
WHERE {
  ?cons jolux:isMemberOf ?base .
  FILTER(STRSTARTS(STR(?base), "https://fedlex.data.admin.ch/eli/cc/"))
  FILTER(REGEX(STR(?cons), "/[0-9]{8}$"))

  # Lois fédérales uniquement
  ?base jolux:typeDocument ?typeDoc .
  VALUES ?typeDoc {
    <https://fedlex.data.admin.ch/vocabulary/resource-type/21>
    <https://fedlex.data.admin.ch/vocabulary/resource-type/22>
    <https://fedlex.data.admin.ch/vocabulary/resource-type/35>
  }

  # "Ce texte est en vigueur" (statut sur le base-act)
  ?base jolux:inForceStatus ?status .
  VALUES ?status {
    <https://fedlex.data.admin.ch/vocabulary/enforcement-status/0>
  }

  OPTIONAL { ?base jolux:title ?title . }
}
"""


    data = client.query_json(sparql)

    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        raise RuntimeError(
            "SPARQL returned 0 rows for CC catalog even after using jolux:isMemberOf. "
            "We need to inspect bindings keys and/or try a broader query."
        )

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
                "title": b.get("title", {}).get("value"),
                "in_force_status_uri": b.get("inForceStatus", {}).get("value"),
                "type_doc": b.get("typeDoc", {}).get("value"),
                "in_force_status_uri": b.get("status", {}).get("value"),    
            }
        )

    df = pd.DataFrame.from_records(records)

    # filtre base-act "propres"
    df = df[df["base_act_uri"].map(lambda x: bool(_BASE_ACT_RE.match(str(x))))].copy()

    df["consolidation_date_yyyymmdd"] = df["consolidation_uri"].map(_extract_last_segment).astype(str)

    df = (
        df.sort_values(["base_act_uri", "consolidation_date_yyyymmdd"])
        .groupby("base_act_uri", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    df["lang"] = "fr"
    df["sr_number"] = None
    return df


