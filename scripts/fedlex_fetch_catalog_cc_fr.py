from __future__ import annotations

from portfolio_repo.fedlex.sparql import SparqlClient
from portfolio_repo.fedlex.catalog import build_cc_catalog_fr
from portfolio_repo.paths import data_dir, ensure_dir


def main() -> int:
    client = SparqlClient()
    df = build_cc_catalog_fr(client)
    print("type_doc value_counts:")
    print(df["type_doc"].value_counts(dropna=False).to_string())


    out_dir = ensure_dir(data_dir("processed") / "fedlex")
    out_path = out_dir / "cc_catalog_fr_latest.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
