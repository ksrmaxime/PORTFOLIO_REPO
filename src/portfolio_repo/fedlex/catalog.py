from __future__ import annotations

from typing import List

import pandas as pd

from portfolio_repo.fedlex.sparql import SparqlClient


def _extract_last_segment(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def build_cc_catalog_fr(client: SparqlClient) -> pd.DataFrame:
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

  # Titre FR (prioritaire)
  OPTIONAL {
    ?base jolux:title ?title_fr .
    FILTER(lang(?title_fr) = "fr")
  }

  # Fallback: un titre quelconque
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

        title_fr = b.get("title_fr", {}).get("value")
        title_any = b.get("title_any", {}).get("value")

        records.append(
            {
                "base_act_uri": base,
                "consolidation_uri": cons,
                "type_doc": b.get("typeDoc", {}).get("value"),
                "in_force_status_uri": b.get("status", {}).get("value"),
                "title_fr": title_fr,
                "title_any": title_any,
            }
        )

    df = pd.DataFrame.from_records(records)

    # Date depuis l’URI de consolidation
    df["consolidation_date_yyyymmdd"] = df["consolidation_uri"].map(_extract_last_segment).astype(str)

    # Conserver le dernier état (dernière consolidation)
    df = (
        df.sort_values(["base_act_uri", "consolidation_date_yyyymmdd"])
        .groupby("base_act_uri", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # IMPORTANT: agrégation titre non-null par base_act_uri (pour éviter de perdre le titre)
    # On récupère le premier titre FR disponible, sinon un titre quelconque.
    title_map = (
        pd.DataFrame.from_records(records)
        .assign(title=lambda x: x["title_fr"].fillna(x["title_any"]))
        .groupby("base_act_uri")["title"]
        .apply(lambda s: next((v for v in s.tolist() if isinstance(v, str) and v.strip()), None))
        .to_dict()
    )
    df["title"] = df["base_act_uri"].map(title_map)

    df["lang"] = "fr"
    return df
