from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from lxml import etree
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


def _norm_title(s: str) -> str:
    # normalise whitespace + remove leading/trailing junk
    return " ".join(s.split()).strip(" \t\r\n-–—")


def extract_title_from_akomantoso_xml(xml_path: Path) -> Optional[str]:
    """
    Best-effort, structure-based extraction of the law title from Fedlex Akoma Ntoso XML.

    Key fix:
    - In many Fedlex XMLs, docTitle contains editorial history in <inline> (e.g. "Nouvelle teneur...").
      We remove inline + editorial nodes before reading the title.
    """
    raw = xml_path.read_bytes()
    if not raw.strip():
        return None

    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        root = etree.fromstring(raw, parser=parser)
    except Exception:
        return None
    if root is None:
        return None

    def norm(s: str) -> str:
        return _norm_title(s)

    def text_of(node: etree._Element) -> str:
        return norm("".join(node.itertext()))

    def cleaned_doc_title_text(doc_title: etree._Element) -> str:
        # Clone node to avoid mutating the parsed tree
        clone = etree.fromstring(etree.tostring(doc_title))
        # Remove typical editorial / revision containers inside titles
        for n in clone.xpath(
            ".//*[local-name()='inline' or local-name()='note' or local-name()='authorialNote' or "
            "local-name()='editorialNote' or local-name()='remark' or local-name()='mod' or "
            "local-name()='quotedStructure' or local-name()='embeddedStructure' or local-name()='commentary']"
        ):
            parent = n.getparent()
            if parent is not None:
                parent.remove(n)

        t = text_of(clone)
        return t

    # Priority 1: docTitle (cleaned)
    doc_titles = root.xpath("//*[local-name()='docTitle']")
    for dt in doc_titles:
        t = cleaned_doc_title_text(dt)
        if t:
            return t

    # Priority 2: shortTitle
    for n in root.xpath("//*[local-name()='shortTitle']"):
        t = text_of(n)
        if t:
            return t

    # Priority 3: FRBRname (Work/Expression/Manifestation)
    nodes = root.xpath(
        "//*[local-name()='FRBRWork']//*[local-name()='FRBRname'] | "
        "//*[local-name()='FRBRExpression']//*[local-name()='FRBRname'] | "
        "//*[local-name()='FRBRManifestation']//*[local-name()='FRBRname']"
    )
    for n in nodes:
        val = n.get("value")
        if val:
            t = norm(val)
            if t:
                return t
        t = text_of(n)
        if t:
            return t

    # Priority 4: very conservative fallback (first preface heading/title)
    nodes = root.xpath("(//*[local-name()='preface']//*[local-name()='heading' or local-name()='title'])[1]")
    for n in nodes:
        t = text_of(n)
        if t:
            return t

    return None


def main() -> int:
    cat_path = data_dir("processed") / "fedlex" / "cc_catalog_fr_latest.csv"
    dl_log_path = data_dir("processed") / "fedlex" / "cc_download_log_fr.csv"

    cat = pd.read_csv(cat_path)
    dl = pd.read_csv(dl_log_path)

    # Normalize date column name in catalog -> cons_date
    if "cons_date" not in cat.columns:
        if "consolidation_date_yyyymmdd" in cat.columns:
            cat = cat.rename(columns={"consolidation_date_yyyymmdd": "cons_date"})
        else:
            raise RuntimeError(f"Catalog has no cons_date column. Columns={list(cat.columns)}")

    cat["base_act_uri"] = cat["base_act_uri"].astype(str)
    cat["cons_date"] = cat["cons_date"].astype(str)

    dl["base_act_uri"] = dl["base_act_uri"].astype(str)
    dl["cons_date"] = dl["cons_date"].astype(str)
    dl["ok_norm"] = dl["ok"].map(is_true)

    df = cat.merge(
        dl[["base_act_uri", "cons_date", "path", "ok_norm"]],
        how="left",
        on=["base_act_uri", "cons_date"],
        validate="1:1",
    )

    df = df[df["ok_norm"] == True].copy()  # noqa: E712

    rows: List[Dict[str, Any]] = []
    err_rows: List[Dict[str, Any]] = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        xml_path = Path(str(r["path"]))
        if not xml_path.exists():
            err_rows.append(
                {
                    "base_act_uri": r["base_act_uri"],
                    "cons_date": r["cons_date"],
                    "error": "Missing XML file",
                    "path": str(xml_path),
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
                    "error": str(e),
                    "path": str(xml_path),
                }
            )
            continue

        # Title strategy:
        # 1) take catalog title if present and non-empty
        # 2) else extract from XML (robust)
        title = r.get("title")
        title = _norm_title(str(title)) if isinstance(title, str) else ""
        if not title:
            xml_title = extract_title_from_akomantoso_xml(xml_path)
            title = xml_title or None

        rows.append(
            {
                "base_act_uri": r.get("base_act_uri"),
                "consolidation_uri": r.get("consolidation_uri"),
                "cons_date": r.get("cons_date"),
                "title": title,
                "type_doc": r.get("type_doc"),
                "clean_text": clean_text,
            }
        )

    out_dir = ensure_dir(data_dir("processed") / "fedlex")
    out_parquet = out_dir / "laws_federal_fr_inforce_clean.parquet"
    out_csv = out_dir / "laws_federal_fr_inforce_clean.csv"
    err_csv = out_dir / "laws_federal_fr_inforce_clean_errors.csv"

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_parquet, index=False)
    out_df.to_csv(out_csv, index=False)
    pd.DataFrame(err_rows).to_csv(err_csv, index=False)

    print(f"Wrote dataset: {out_parquet} ({len(out_df)} rows)")
    print(f"Wrote CSV:     {out_csv}")
    print(f"Errors: {len(err_rows)} -> {err_csv}")

    # Quick sanity check: missing titles
    if "title" in out_df.columns:
        missing = int(out_df["title"].isna().sum() + (out_df["title"].astype(str).str.strip() == "").sum())
        print(f"Missing titles (after XML fallback): {missing}/{len(out_df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
