from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.paths import find_repo_root
from portfolio_repo.fedlex.download_xml_flat import download_many_xml_flat


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Fedlex XML into a flat directory (data/raw/fedlex/xml).")
    ap.add_argument("--registry", required=True, help="Path to a registry parquet/csv containing law_id, xml_url, lang.")
    ap.add_argument("--overwrite", action="store_true", help="Re-download even if file exists.")
    args = ap.parse_args()

    repo = find_repo_root()
    reg_path = Path(args.registry)
    if not reg_path.is_absolute():
        reg_path = repo / reg_path

    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")

    if reg_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(reg_path)
    elif reg_path.suffix.lower() == ".csv":
        df = pd.read_csv(reg_path)
    else:
        raise ValueError("Registry must be .parquet or .csv")

    # expected cols: law_id, xml_url, lang
    log_path = download_many_xml_flat(df, overwrite=bool(args.overwrite))
    print(f"Wrote download log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
