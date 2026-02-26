#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


# === LISTE DES LOIS À CONSERVER ===
KEEP_LAWS = {
    "https://fedlex.data.admin.ch/eli/cc/1959/679_705_685",
    "https://fedlex.data.admin.ch/eli/cc/1993/1798_1798_1798",
    "https://fedlex.data.admin.ch/eli/cc/1997/2187_2187_2187",
    "https://fedlex.data.admin.ch/eli/cc/1998/2535_2535_2535",
    "https://fedlex.data.admin.ch/eli/cc/2002/226",
    "https://fedlex.data.admin.ch/eli/cc/2007/150",
    "https://fedlex.data.admin.ch/eli/cc/2017/494",
    "https://fedlex.data.admin.ch/eli/cc/2022/232",
    "https://fedlex.data.admin.ch/eli/cc/2022/491",
    "https://fedlex.data.admin.ch/eli/cc/2022/537",
}


def main():
    input_path = Path("data/processed/fedlex/all_in_one/laws_structure_final.parquet")
    output_parquet = input_path.with_name("laws_structure_selected.parquet")
    output_csv = input_path.with_name("laws_structure_selected.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading parquet…")
    df = pd.read_parquet(input_path)

    print("Filtering laws…")
    df_filtered = df[df["law_id"].isin(KEEP_LAWS)].copy()

    print(f"Rows before: {len(df)}")
    print(f"Rows after:  {len(df_filtered)}")

    print("Saving parquet…")
    df_filtered.to_parquet(output_parquet, index=False)

    print("Saving CSV (Excel-friendly)…")
    df_filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Parquet: {output_parquet}")
    print(f"CSV:     {output_csv}")


if __name__ == "__main__":
    main()
