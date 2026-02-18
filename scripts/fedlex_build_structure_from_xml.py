from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_repo.fedlex.structure_from_xml import build_structure_from_catalog


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Path to cc catalog parquet (contains base_act_uri, title, xml filename/uri)")
    ap.add_argument("--xml-dir", required=True, help="Directory containing downloaded XML files")
    ap.add_argument("--out", required=True, help="Output parquet path (laws_structure.parquet)")
    args = ap.parse_args()

    build_structure_from_catalog(
        catalog_parquet=Path(args.catalog),
        xml_dir=Path(args.xml_dir),
        out_parquet=Path(args.out),
    )
    print(f"Saved structure parquet to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
