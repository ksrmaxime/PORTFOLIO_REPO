from __future__ import annotations

import argparse
import requests
import pandas as pd

ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"

def run_query(q: str) -> pd.DataFrame:
    r = requests.post(
        ENDPOINT,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"query": q},
        timeout=60,
    )
    r.raise_for_status()
    bindings = r.json().get("results", {}).get("bindings", [])
    rows = []
    for b in bindings:
        rows.append({k: v.get("value") for k, v in b.items()})
    return pd.DataFrame(rows)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", required=True, help="A CC URI like https://fedlex.data.admin.ch/eli/cc/2025/620")
    args = ap.parse_args()

    cc = args.cc

    q1 = f"""
SELECT ?p ?o
WHERE {{
  <{cc}> ?p ?o .
}}
LIMIT 200
"""
    df1 = run_query(q1)
    print("\n=== Direct triples on CC ===")
    print(df1.to_string(index=False))

    q2 = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
SELECT ?work ?p ?o
WHERE {{
  ?work jolux:isMemberOf <{cc}> ;
        ?p ?o .
}}
LIMIT 300
"""
    df2 = run_query(q2)
    print("\n=== Triples on member work(s) (jolux:isMemberOf CC) ===")
    print(df2.to_string(index=False))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
