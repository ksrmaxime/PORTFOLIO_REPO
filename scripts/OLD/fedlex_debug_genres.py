from __future__ import annotations

import pandas as pd
import requests

ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"
SCHEME = "https://fedlex.data.admin.ch/vocabulary/legal-resource-genre"

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
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"query": SPARQL},
        timeout=60,
    )
    r.raise_for_status()
    bindings = r.json()["results"]["bindings"]
    rows = []
    for b in bindings:
        rows.append(
            {
                "term": b["term"]["value"],
                "label": b["label"]["value"],
                "lang": b["lang"]["value"],
            }
        )

    df = pd.DataFrame(rows)

    # Show likely candidates for "Federal Act" and likely ordinances
    df_de = df[df["lang"].isin(["de", "fr", "en"])].copy()
    candidates = df_de[
        df_de["label"].str.contains("bundesgesetz|loi féd|federal act|act", case=False, regex=True)
        | df_de["label"].str.contains("verordnung|ordonnance|ordinance", case=False, regex=True)
    ].copy()

    print("=== Candidate genres (acts + ordinances) ===")
    print(candidates.sort_values(["label", "lang"]).to_string(index=False))
    print("\n=== Total concepts ===", df["term"].nunique())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
