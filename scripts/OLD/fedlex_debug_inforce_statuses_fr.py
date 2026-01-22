from __future__ import annotations

import pandas as pd

from portfolio_repo.fedlex.sparql import SparqlClient


def main() -> int:
    client = SparqlClient()
    q = """
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
PREFIX skos:  <http://www.w3.org/2004/02/skos/core#>

SELECT ?status ?label (COUNT(?base) AS ?n)
WHERE {
  ?base jolux:typeDocument ?typeDoc .
  VALUES ?typeDoc {
    <https://fedlex.data.admin.ch/vocabulary/resource-type/21>
    <https://fedlex.data.admin.ch/vocabulary/resource-type/22>
  }
  OPTIONAL {
    ?base jolux:inForceStatus ?status .
    OPTIONAL { ?status skos:prefLabel ?label . FILTER(lang(?label)="fr") }
  }
}
GROUP BY ?status ?label
ORDER BY DESC(?n)
"""
    js = client.query_json(q)
    rows = []
    for b in js["results"]["bindings"]:
        rows.append(
            {
                "status": b.get("status", {}).get("value"),
                "label": b.get("label", {}).get("value"),
                "n": int(b["n"]["value"]),
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
