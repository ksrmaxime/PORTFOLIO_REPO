from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import requests

from portfolio_repo.paths import project_paths


# ------------------------------------------------------------------
# FEDLEX CC COLLECTION (JSON)
# ------------------------------------------------------------------
CC_COLLECTION_URL = "https://fedlex.data.admin.ch/eli/cc?format=json"


def _yyyymmdd(x) -> str:
    return pd.to_datetime(x).strftime("%Y%m%d")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build Fedlex CC registry (FR only) from ELI collection."
    )
    ap.add_argument("--repo-root", type=str, default=None)
    ap.add_argument(
        "--out",
        type=str,
        default="data/processed/fedlex_registry.parquet",
        help="Output registry parquet (relative to repo root).",
    )
    args = ap.parse_args()

    paths = project_paths(args.repo_root)

    print("Downloading Fedlex CC collection…")

    r = requests.get(CC_COLLECTION_URL, timeout=60)
    r.raise_for_status()
    payload = r.json()

    members = payload.get("members")
    if not members:
        raise RuntimeError("Fedlex CC collection returned no members.")

    rows = []
    for item in members:
        eli_uri = item.get("uri")
        if not eli_uri:
            continue

        titles = item.get("title", {})
        title_fr = titles.get("fr")
        if not title_fr:
            continue

        date_entry = item.get("dateEntryInForce")
        if not date_entry:
            continue

        rows.append(
            {
                "eli_uri": eli_uri,
                "title": title_fr,
                "title_lang": "fr",
                "date_entry_in_force": date_entry,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No usable CC laws found in Fedlex collection.")

    # ------------------------------------------------------------------
    # BUILD XML URL
    # ------------------------------------------------------------------
    df["xml_url"] = (
        df["eli_uri"].astype(str)
        + "/"
        + df["date_entry_in_force"].apply(_yyyymmdd)
        + "/fr/xml"
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = paths.root / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Registry written to: {out_path}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())







