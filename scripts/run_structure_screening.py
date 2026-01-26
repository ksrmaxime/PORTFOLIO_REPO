from __future__ import annotations

import argparse
import pandas as pd

from portfolio_repo.llm.client import LocalLLMClient, LLMConfig
from portfolio_repo.llm.structure_screening import (
    screen_law_structure,
    StructureScreeningConfig,
)


# ============================================================
# Script
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="RUN 1 — Structure-only LLM screening (labels only)"
    )
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--max_laws", type=int, default=0)
    ap.add_argument("--base_url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="apertus-local")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)

    if args.max_laws > 0:
        keep = df["law_id"].drop_duplicates().head(args.max_laws)
        df = df[df["law_id"].isin(keep)].copy()

    client = LocalLLMClient(
        LLMConfig(
            base_url=args.base_url,
            model=args.model,
        )
    )

    cfg = StructureScreeningConfig(
        debug=args.debug,
    )

    # Colonnes de sortie
    df["relevance"] = False
    df["relevance_reason"] = ""

    # ========================================================
    # Boucle par loi
    # ========================================================
    for idx, (law_id, g) in enumerate(df.groupby("law_id", sort=False), start=1):
        print(f"[{idx}] Screening law")

        node_reason_map = screen_law_structure(
            client=client,
            law_numeric_id=idx,
            structure_df=g,
            cfg=cfg,
        )

        if not node_reason_map:
            continue

        # Propagation descendante
        children = {}
        for row in g.itertuples():
            if pd.notna(row.parent_node_id):
                children.setdefault(row.parent_node_id, []).append(row.node_id)

        stack = list(node_reason_map.keys())
        seen = set()

        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(children.get(cur, []))

        # Marquage relevance
        df.loc[df["node_id"].isin(seen), "relevance"] = True

        # Justification UNIQUEMENT sur les noeuds sélectionnés par le LLM
        for node_id, reason in node_reason_map.items():
            df.loc[df["node_id"] == node_id, "relevance_reason"] = reason

    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {args.out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
