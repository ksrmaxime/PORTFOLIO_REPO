"""Filtre le dataset Fedlex (laws_structure_final) aux actes dont au moins un article
ou un titre de partie/chapitre/section contient un mot-clé (cf. src/keyword_filter.py).

Sorties (à côté de --output_base) :
  <base>.parquet / .csv            toutes les lignes des actes retenus (+ colonnes kw_*, act_*)
  <base>_acts.csv                  1 ligne par acte retenu : RS, titre, type, mots-clés trouvés
  <base>_keywords.csv              nb de lignes / d'actes par mot-clé (pour juger chaque motif)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse

import pandas as pd

from src.keyword_filter import KEYWORDS, compile_keywords, filter_dataframe


def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"label": str, "text": str, "sr_number": str}, keep_default_na=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="laws_structure_final.parquet (ou .csv)")
    ap.add_argument("--output_base", required=True, help="Chemin de sortie sans extension")
    ap.add_argument("--tiers", default="core,extended", help="Niveaux de mots-clés : core, extended (défaut : les deux)")
    ap.add_argument("--groups", default="", help="Restreindre à certains groupes (séparés par des virgules)")
    ap.add_argument("--levels", default="1,2,3,4,5,6", help="Niveaux de structure fouillés (défaut : 1-6, titre de l'acte exclu)")
    args = ap.parse_args()

    tiers = [t for t in args.tiers.split(",") if t]
    groups = [g for g in args.groups.split(",") if g] or None
    levels = [int(x) for x in args.levels.split(",") if x]

    df = read_any(args.input)
    print(f"Lu : {args.input} ({len(df):,} lignes, {df['law_id'].nunique():,} actes)")

    compiled = compile_keywords(KEYWORDS, tiers=tiers, groups=groups)
    print(f"Mots-clés : {len(compiled)} motifs (tiers={tiers}, groups={groups or 'tous'}) ; niveaux fouillés : {levels}")

    res = filter_dataframe(df, compiled, levels=levels)
    rows, acts, kw = res["rows"], res["acts"], res["keywords"]

    base = Path(args.output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(f"{base}.parquet", index=False)
    rows.to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
    acts.to_csv(f"{base}_acts.csv", index=False, encoding="utf-8-sig")
    kw.to_csv(f"{base}_keywords.csv", index=False, encoding="utf-8-sig")

    print("\n=== BILAN ===")
    if acts.empty:
        print("Aucun acte retenu.")
        return 0
    print(f"Actes retenus : {len(acts):,} / {df['law_id'].nunique():,} "
          f"(core : {int((acts['act_tier'] == 'core').sum()):,} | extended seulement : {int((acts['act_tier'] == 'extended').sum()):,})")
    print(f"Lignes en sortie : {len(rows):,} (dont lignes contenant un mot-clé : {int(rows['kw_hit'].sum()):,})")
    if "doc_category" in acts.columns:
        print("Par catégorie :", acts["doc_category"].value_counts().to_dict())
    print("\nPar mot-clé :")
    print(kw[["tier", "group", "keyword", "n_acts", "n_rows"]].to_string(index=False))
    print(f"\nSaved: {base}.parquet / .csv, {base}_acts.csv, {base}_keywords.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
