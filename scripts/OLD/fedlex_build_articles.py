from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.paths import project_paths
from portfolio_repo.fedlex.chunking import xml_to_plain_text
from portfolio_repo.fedlex.cleaning import CleaningConfig, clean_fedlex_text
from portfolio_repo.fedlex.articles import split_by_article


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--registry", default="fedlex_registry.parquet")
    ap.add_argument("--download-log", default="fedlex_download_log.parquet")
    ap.add_argument("--out", default="fedlex_articles.parquet")
    ap.add_argument("--lang", default="fr", help="Language to keep (default: fr)")
    ap.add_argument("--only-ok", action="store_true", help="Use only OK downloads from the log")
    ap.add_argument("--include-preamble", action="store_true", help="Also output a PREAMBLE row per law")
    args = ap.parse_args()

    paths = project_paths(args.repo_root)

    reg = pd.read_parquet(paths.data_processed / args.registry)
    log = pd.read_parquet(paths.data_processed / args.download_log)

    if "eli_uri" not in reg.columns:
        raise RuntimeError("Registry missing 'eli_uri'.")
    if "eli_uri" not in log.columns or "xml_path" not in log.columns:
        raise RuntimeError("Download log missing required columns.")

    lang = args.lang.strip().lower()

    if args.only_ok:
        if "ok" not in log.columns:
            raise RuntimeError("Download log missing 'ok' column; cannot use --only-ok.")
        log = log[log["ok"] == True].copy()

    log = log[log["lang"].astype(str).str.lower() == lang].copy()
    log = log[log["xml_path"].notna()].copy()
    if log.empty:
        raise RuntimeError(f"No XML paths for lang={lang} found in download log.")

    # Stable ID
    reg = reg.copy()
    reg["law_id"] = reg["eli_uri"].astype(str)

    log = log.copy()
    log["eli_uri"] = log["eli_uri"].astype(str)
    log["law_id"] = log["eli_uri"]

    # Merge metadata (avoid eli_uri duplicate)
    reg_meta_cols = [c for c in ["law_id", "title", "title_lang", "type_uri", "date_entry_in_force"] if c in reg.columns]
    reg_small = reg[reg_meta_cols].drop_duplicates("law_id")
    log_small = log[["law_id", "eli_uri", "lang", "xml_path", "sha256"]].copy()
    merged = log_small.merge(reg_small, on="law_id", how="left")

    cfg = CleaningConfig()

    rows: list[dict] = []

    for r in merged.itertuples(index=False):
        xml_path = Path(r.xml_path)
        if not xml_path.exists():
            continue

        xml_bytes = xml_path.read_bytes()
        raw = xml_to_plain_text(xml_bytes, apply_cleaning=False)
        clean = clean_fedlex_text(raw, cfg=cfg)

        preamble, arts = split_by_article(clean)

        if args.include_preamble and preamble:
            rows.append(
                {
                    "law_id": r.law_id,
                    "eli_uri": r.eli_uri,
                    "lang": r.lang,
                    "section_type": "PREAMBLE",
                    "art_id": None,
                    "art_number": None,
                    "art_suffix": None,
                    "text_raw": raw,
                    "text_clean": clean,
                    "section_text_clean": preamble,
                    "xml_path": str(xml_path),
                    "sha256": getattr(r, "sha256", None),
                    "title": getattr(r, "title", None),
                    "title_lang": getattr(r, "title_lang", None),
                    "type_uri": getattr(r, "type_uri", None),
                    "date_entry_in_force": getattr(r, "date_entry_in_force", None),
                }
            )

        for a in arts:
            rows.append(
                {
                    "law_id": r.law_id,
                    "eli_uri": r.eli_uri,
                    "lang": r.lang,
                    "section_type": "ARTICLE",
                    "art_id": a.art_id,
                    "art_number": a.art_number,
                    "art_suffix": a.art_suffix,
                    "text_raw": raw,
                    "text_clean": clean,
                    "section_text_clean": a.text,
                    "xml_path": str(xml_path),
                    "sha256": getattr(r, "sha256", None),
                    "title": getattr(r, "title", None),
                    "title_lang": getattr(r, "title_lang", None),
                    "type_uri": getattr(r, "type_uri", None),
                    "date_entry_in_force": getattr(r, "date_entry_in_force", None),
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = paths.data_processed / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    n_laws = out_df["law_id"].nunique() if not out_df.empty else 0
    n_articles = (out_df["section_type"] == "ARTICLE").sum() if "section_type" in out_df.columns else 0
    print(f"Wrote articles: {out_path} ({len(out_df):,} rows, {n_laws} laws, {n_articles} articles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
