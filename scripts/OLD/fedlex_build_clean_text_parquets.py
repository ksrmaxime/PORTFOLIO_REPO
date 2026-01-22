from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_repo.fedlex.io_clean_parquet import build_clean_parquets_from_xml_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Build clean law-level and article-level parquet files from AKN XML.")
    ap.add_argument("--xml-dir", type=str, required=True, help="Directory containing downloaded Fedlex AKN XML files.")
    ap.add_argument("--glob", type=str, default="*.xml", help="Glob pattern for XML files (default: *.xml).")
    ap.add_argument("--law-id-from-filename", action="store_true", help="Use filename stem as law_id (recommended).")
    args = ap.parse_args()

    # We keep config simple in the script; tweak in code if you want.
    laws_path, arts_path = build_clean_parquets_from_xml_dir(
        xml_dir=Path(args.xml_dir),
        law_id_from_filename=bool(args.law_id_from_filename),
    )

    print(f"Wrote: {laws_path}")
    print(f"Wrote: {arts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
