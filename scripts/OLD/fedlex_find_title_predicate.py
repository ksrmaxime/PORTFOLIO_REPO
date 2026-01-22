from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Optional

import pandas as pd
import requests

ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"


def is_title_like(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 8:
        return False
    # Exclude obvious dates/ids
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return False
    # Titles usually have spaces and letters
    if sum(c.isalpha() for c in s) < 6:
        return False
    return True


def run_query(q: str) -> list[dict]:
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
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="data/processed/fedlex_registry.parquet")
    ap.add_argument("--n", type=int, default=25, help="How many CCs to sample")
    ap.add_argument("--lang", default="fr")
    args = ap.parse_args()

    df = pd.read_parquet(args.registry)
    if df.empty:
        raise SystemExit("Registry is empty.")

    lang = args.lang.strip().lower()
    ccs = df["eli_uri"].dropna().astype(str).head(args.n).tolist()

    pred_counts = Counter()
    examples: dict[str, str] = {}

    for cc in ccs:
        expr = f"{cc}/{lang}"
        q = f"""
SELECT ?p ?o
WHERE {{
  <{expr}> ?p ?o .
  FILTER(isLiteral(?o))
  FILTER(lang(?o) = "{lang}")
}}
LIMIT 200
"""
        rows = run_query(q)
        for r in rows:
            p = r.get("p")
            o = r.get("o")
            if not p or not o:
                continue
            if is_title_like(o):
                pred_counts[p] += 1
                examples.setdefault(p, o)

    if not pred_counts:
        print("No French literal candidates found on sampled /fr expressions.")
        print("This likely means the title is NOT stored directly on <cc>/fr, but via another linked node.")
        return 0

    print("=== Candidate title predicates (ranked) ===")
    for p, cnt in pred_counts.most_common(30):
        ex = examples.get(p, "")[:120].replace("\n", " ")
        print(f"{cnt:>3}  {p}\n     e.g. {ex}\n")

    best = pred_counts.most_common(1)[0][0]
    print("BEST_GUESS_PREDICATE =", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
