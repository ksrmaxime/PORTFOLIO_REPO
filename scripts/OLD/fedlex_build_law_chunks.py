# scripts/fedlex_build_law_chunks.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_repo.paths import project_paths
from portfolio_repo.fedlex.chunking import chunk_text, xml_to_plain_text


def _chunk_to_text(ch) -> str:
    # chunk_text() may return either strings or Chunk objects (dataclass).
    if isinstance(ch, str):
        return ch
    txt = getattr(ch, "text", None)
    if isinstance(txt, str):
        return txt
    # last resort: string representation (should not happen)
    return str(ch)


def _chunk_to_index(ch, fallback_index: int) -> int:
    idx = getattr(ch, "chunk_index", None)
    if isinstance(idx, int):
        return idx
    return fallback_index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--registry", default="fedlex_registry.parquet")
    ap.add_argument("--download-log", default="fedlex_download_log.parquet")
    ap.add_argument("--out", default="fedlex_law_chunks.parquet")
    ap.add_argument("--target-chars", type=int, default=10_000)
    ap.add_argument("--overlap-chars", type=int, default=300)
    ap.add_argument("--only-ok", action="store_true")
    args = ap.parse_args()

    paths = project_paths(args.repo_root)

    reg_path = paths.data_processed / args.registry
    log_path = paths.data_processed / args.download_log

    reg = pd.read_parquet(reg_path)
    log = pd.read_parquet(log_path)

    if "eli_uri" not in reg.columns:
        raise RuntimeError(f"Registry missing 'eli_uri': {reg_path}")
    if "eli_uri" not in log.columns or "xml_path" not in log.columns:
        raise RuntimeError(f"Download log missing required columns: {log_path}")

    if args.only_ok:
        if "ok" not in log.columns:
            raise RuntimeError("Download log missing 'ok' column; cannot use --only-ok.")
        log = log[log["ok"] == True].copy()

    log = log[log["xml_path"].notna()].copy()
    if log.empty:
        raise RuntimeError("No XML paths found in download log. Run fedlex_download_xml.py first.")

    # Stable IDs
    reg = reg.copy()
    reg["law_id"] = reg["eli_uri"].astype(str)

    log = log.copy()
    log["eli_uri"] = log["eli_uri"].astype(str)
    log["law_id"] = log["eli_uri"]

    # Keep log columns we need
    log_keep_cols = ["law_id", "eli_uri", "lang", "xml_path", "sha256"]
    for c in log_keep_cols:
        if c not in log.columns:
            if c == "sha256":
                log[c] = None
            else:
                raise RuntimeError(f"Download log missing column '{c}'")
    log_small = log[log_keep_cols].copy()

    # Keep registry metadata (no eli_uri to avoid merge collisions)
    reg_meta_cols = [c for c in ["law_id", "title", "title_lang", "type_uri", "date_entry_in_force"] if c in reg.columns]
    reg_small = reg[reg_meta_cols].drop_duplicates("law_id")

    merged = log_small.merge(reg_small, on="law_id", how="left")

    rows = []
    for r in merged.itertuples(index=False):
        xml_path = Path(r.xml_path)
        if not xml_path.exists():
            continue

        xml_bytes = xml_path.read_bytes()
        plain = xml_to_plain_text(xml_bytes)

        chunks = chunk_text(
            plain,
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
        )

        for i, ch in enumerate(chunks):
            rows.append(
                {
                    "law_id": r.law_id,
                    "eli_uri": r.eli_uri,
                    "lang": r.lang,
                    "chunk_index": _chunk_to_index(ch, i),
                    "chunk_text": _chunk_to_text(ch),  # <-- critical fix
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

    print(f"Wrote law chunks: {out_path} ({len(out_df):,} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



