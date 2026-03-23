from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from score_src import ColumnConfig, compare_frames, save_eval_outputs


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {path}")


def parse_mapping_arg(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid mapping item '{part}'. Expected key=value.")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_column_configs(
    cols: list[str],
    *,
    kinds_map: dict[str, str],
    list_sep_map: dict[str, str],
    drop_na_pairs: bool,
) -> list[ColumnConfig]:
    cfgs: list[ColumnConfig] = []
    for c in cols:
        kind = kinds_map.get(c, "label")
        list_sep = list_sep_map.get(c, "|")
        cfgs.append(
            ColumnConfig(
                name=c,
                kind=kind,
                drop_na_pairs=drop_na_pairs,
                list_sep=list_sep,
            )
        )
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pred", required=True, help="Prediction file (csv/parquet)")
    ap.add_argument("--gold", required=True, help="Gold/reference file (csv/parquet)")
    ap.add_argument("--cols", required=True, help="Comma-separated columns to compare")
    ap.add_argument("--id_col", help="Join key. Not needed with --use_row_order")
    ap.add_argument("--use_row_order", action="store_true", help="Compare by row order instead of explicit ID")
    ap.add_argument("--max_rows", type=int, default=None, help="Keep only the first N rows from each file")
    ap.add_argument("--keep_na", action="store_true", help="Keep NA-vs-NA / NA-vs-value pairs in evaluation")
    ap.add_argument("--col_kinds", default="", help="Per-column kinds, e.g. RELEVANT_ART=label,topics=list")
    ap.add_argument("--list_seps", default="", help="Per-column list separators, e.g. topics=|,targets=;")
    ap.add_argument("--report_dir", default=None, help="Directory where evaluation reports will be saved")
    ap.add_argument("--print_errors_head", type=int, default=10, help="How many disagreement rows to preview per column")
    ap.add_argument(
        "--invert_gold_cols",
        default="",
        help="Comma-separated columns whose YES/NO gold values should be flipped before scoring",
    )
    ap.add_argument(
        "--rename_gold_cols",
        default="",
        help="Rename gold columns before scoring, e.g. old_name=new_name",
    )

    args = ap.parse_args()

    pred = read_any(args.pred)
    gold = read_any(args.gold)

    if args.max_rows is not None:
        pred = pred.head(args.max_rows).reset_index(drop=True)
        gold = gold.head(args.max_rows).reset_index(drop=True)

    rename_gold_map = parse_mapping_arg(args.rename_gold_cols)
    if rename_gold_map:
        gold = gold.rename(columns=rename_gold_map)

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    if not cols:
        raise ValueError("No columns provided in --cols")

    if args.use_row_order:
        pred = pred.reset_index(drop=True).copy()
        gold = gold.reset_index(drop=True).copy()
        pred["__row__"] = pred.index.astype(int)
        gold["__row__"] = gold.index.astype(int)
        id_col = "__row__"
    else:
        if not args.id_col:
            raise ValueError("You must provide --id_col unless --use_row_order is set.")
        id_col = args.id_col

    invert_gold_cols = {c.strip() for c in args.invert_gold_cols.split(",") if c.strip()}
    if invert_gold_cols:
        _flip = {"YES": "NO", "NO": "YES", "yes": "no", "no": "yes", "Yes": "No", "No": "Yes"}
        for c in invert_gold_cols:
            if c in gold.columns:
                gold[c] = gold[c].map(lambda v: _flip.get(str(v).strip(), v) if not pd.isna(v) else v)

    kinds_map = parse_mapping_arg(args.col_kinds)
    list_sep_map = parse_mapping_arg(args.list_seps)
    column_configs = build_column_configs(
        cols,
        kinds_map=kinds_map,
        list_sep_map=list_sep_map,
        drop_na_pairs=(not args.keep_na),
    )

    result, merged, confusion_tables, classification_reports, label_distributions = compare_frames(
        pred=pred,
        gold=gold,
        id_col=id_col,
        column_configs=column_configs,
    )

    print("=== EVAL ===")
    print(f"Joined rows: {result.n_joined_rows:,}")
    print(f"Compared cells: {result.n_compared_total:,}")
    print(f"Equal cells: {result.n_equal_total:,}")
    print(f"Similarity: {result.similarity_pct:.2f}%")

    for c in cols:
        r = result.columns[c]
        print(f"\n--- {c} ({r.kind}) ---")
        print(f"Compared: {r.n_compared:,}")
        print(f"Equal: {r.n_equal:,}")
        print(f"Similarity: {r.similarity_pct:.2f}%")
        print(
            f"Missing -> pred: {r.n_missing_pred:,} | gold: {r.n_missing_gold:,} | "
            f"both: {r.n_missing_both:,} | either: {r.n_missing_either:,}"
        )

        if r.macro_f1 is not None:
            print(
                f"Macro Precision: {r.macro_precision:.4f} | "
                f"Macro Recall: {r.macro_recall:.4f} | "
                f"Macro F1: {r.macro_f1:.4f} | "
                f"Balanced Acc: {r.balanced_accuracy:.4f}"
            )

        if r.avg_jaccard is not None:
            print(f"Average Jaccard: {r.avg_jaccard:.4f}")

        err_path_df = merged.loc[
            merged[f"{c}__compared"].fillna(False) & (merged[f'{c}__match'] == False),
            [id_col, f"{c}__pred_norm", f"{c}__gold_norm"],
        ].head(args.print_errors_head)

        if not err_path_df.empty:
            print("Example mismatches:")
            print(err_path_df.to_string(index=False))

    if args.report_dir:
        save_eval_outputs(
            out_dir=args.report_dir,
            result=result,
            merged=merged,
            confusion_tables=confusion_tables,
            classification_reports=classification_reports,
            label_distributions=label_distributions,
            column_configs=column_configs,
            id_col=id_col,
        )
        print(f"\nReports saved to: {args.report_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
