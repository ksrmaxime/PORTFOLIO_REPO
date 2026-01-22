from __future__ import annotations

import requests
import pandas as pd

ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"
SCHEME = "https://fedlex.data.admin.ch/vocabulary/resource-type"

SPARQL = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?term ?label ?lang
WHERE {{
  ?term a skos:Concept ;
        skos:inScheme <{SCHEME}> ;
        skos:prefLabel ?label .
  BIND(LANG(?label) AS ?lang)
}}
ORDER BY ?term ?lang
"""

def main() -> int:
  r = requests.post(
      ENDPOINT,
      headers={"Accept":"application/sparql-results+json","Content-Type":"application/x-www-form-urlencoded"},
      data={"query": SPARQL},
      timeout=60,
  )
  r.raise_for_status()
  bindings = r.json()["results"]["bindings"]
  rows = [{"term": b["term"]["value"], "label": b["label"]["value"], "lang": b["lang"]["value"]} for b in bindings]
  df = pd.DataFrame(rows)

  cand = df[df["label"].str.contains(
      "loi|gesetz|act|ordonnance|verordnung|ordinance",
      case=False, regex=True
  )].copy()

  print(cand.sort_values(["label","lang"]).to_string(index=False))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
