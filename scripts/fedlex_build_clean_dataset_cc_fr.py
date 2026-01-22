from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from portfolio_repo.fedlex.cleaner import clean_akoma_ntoso_xml_to_text
from portfolio_repo.paths import data_dir, ensure_dir


def is_true(x: Any) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def main() -> int:
    cat_path = data_dir("processed") / "fedlex" / "cc_catalog_fr_latest.csv"
    dl_log_path = data_dir("processed") / "fedlex" / "cc_download_log_fr.csv"

    if not cat_path.exists():
        raise FileNotFoundError(f"Missing catalog: {cat_path}")
    if not dl_log_path.exists():
        raise FileNotFoundError(f"Missing download log: {dl_log_path}")

    cat = pd.read_csv(cat_path)
    dl = pd.read_csv(dl_log_path)

    # ---- Normalize date column name in catalog -> cons_date
    if "cons_date" not in cat.columns:
        if "consolidation_date_yyyymmdd" in cat.columns:
            cat = cat.rename(columns={"consolidation_date_yyyymmdd": "cons_date"})
        else:
            raise RuntimeError(
                f"Catalog missing cons_date (and no consolidation_date_yyyymmdd fallback). "
                f"Catalog columns={list(cat.columns)}"
            )

    # ---- Ensure merge keys are comparable (strings)
    cat["base_act_uri"] = cat["base_act_uri"].astype(str)
    cat["cons_date"] = cat["cons_date"].astype(str)

    dl["base_act_uri"] = dl["base_act_uri"].astype(str)
    if "cons_date" not in dl.columns:
        raise RuntimeError(f"Download log missing cons_date. Columns={list(dl.columns)}")
    dl["cons_date"] = dl["cons_date"].astype(str)

    # ---- Merge
    keep_dl_cols = [c for c in ["base_act_uri", "cons_date", "path", "ok", "url", "http_status", "error"] if c in dl.columns]
    df = cat.merge(
        dl[keep_dl_cols],
        how="left",
        on=["base_act_uri", "cons_date"],
        validate="1:1",
    )

    out_dir = ensure_dir(data_dir("processed") / "fedlex")

    # ---- Diagnostics (write merged debug snapshot)
    debug_path = out_dir / "laws_federal_fr_inforce_merged_debug.csv"
    df.to_csv(debug_path, index=False)

    print(f"[DEBUG] Catalog rows: {len(cat)}")
    print(f"[DEBUG] Download log rows: {len(dl)}")
    print(f"[DEBUG] Rows after merge: {len(df)}")
    print(f"[DEBUG] Merged debug snapshot: {debug_path}")

    if "ok" not in df.columns:
        print("[DEBUG] No 'ok' column after merge -> cannot filter, dataset will be empty.")
        return 1

    print("[DEBUG] ok value_counts (raw):")
    print(df["ok"].astype(str).value_counts(dropna=False).head(20).to_string())

    df["ok_norm"] = df["ok"].map(is_true)
    ok_n = int(df["ok_norm"].sum())
    print(f"[DEBUG] Rows with ok_norm=True: {ok_n}")

    df_ok = df[df["ok_norm"]].copy()
    if "path" not in df_ok.columns:
        print("[DEBUG] No 'path' column after merge -> cannot read XML.")
        return 1

    missing_path = int(df_ok["path"].isna().sum())
    print(f"[DEBUG] Rows ok_norm=True but missing path: {missing_path}")

    # ---- Build dataset
    rows: List[Dict[str, Any]] = []
    err_rows: List[Dict[str, Any]] = []

    for _, r in tqdm(df_ok.iterrows(), total=len(df_ok)):
        xml_path = Path(str(r["path"]))
        if not xml_path.exists():
            err_rows.append(
                {
                    "base_act_uri": r["base_act_uri"],
                    "cons_date": r["cons_date"],
                    "path": str(xml_path),
                    "error": "XML path does not exist on disk",
                }
            )
            continue

        try:
            clean_text = clean_akoma_ntoso_xml_to_text(xml_path)
        except Exception as e:
            err_rows.append(
                {
                    "base_act_uri": r["base_act_uri"],
                    "cons_date": r["cons_date"],
                    "path": str(xml_path),
                    "error": str(e),
                }
            )
            continue

        # Metadata fields (all optional except base_act_uri/cons_date)
        rows.append(
            {
                "base_act_uri": r.get("base_act_uri"),
                "consolidation_uri": r.get("consolidation_uri"),
                "cons_date": r.get("cons_date"),
                "lang": r.get("lang", "fr"),
                "type_doc": r.get("type_doc"),
                "in_force_status_uri": r.get("in_force_status_uri"),
                "title": r.get("title"),
                "download_url": r.get("url"),
                "xml_path": str(xml_path),
                "clean_text": clean_text,
            }
        )

    out_df = pd.DataFrame(rows)
    err_df = pd.DataFrame(err_rows)

    out_parquet = out_dir / "laws_federal_fr_inforce_clean.parquet"
    out_csv = out_dir / "laws_federal_fr_inforce_clean.csv"
    err_csv = out_dir / "laws_federal_fr_inforce_clean_errors.csv"

    out_df.to_parquet(out_parquet, index=False)
    out_df.to_csv(out_csv, index=False)
    err_df.to_csv(err_csv, index=False)

    print(f"Wrote dataset: {out_parquet} ({len(out_df)} rows)")
    print(f"Wrote CSV:     {out_csv}")
    print(f"Errors: {len(err_df)} -> {err_csv}")

    # Hard fail if empty so you notice immediately
    if len(out_df) == 0:
        print("Dataset is EMPTY. Check:")
        print(f" - merged debug: {debug_path}")
        print(f" - errors:       {err_csv}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

