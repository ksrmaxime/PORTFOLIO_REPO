from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import ast
import json
import math
import re

import pandas as pd


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class ColumnConfig:
    name: str
    kind: str = "label"          # label | list | text | number | boolean
    drop_na_pairs: bool = True
    list_sep: str = "|"
    list_as_set: bool = True
    sort_list_values: bool = True
    casefold: bool = True
    strip: bool = True
    collapse_spaces: bool = True


@dataclass(frozen=True)
class ColumnEvalResult:
    column: str
    kind: str
    n_joined_rows: int
    n_compared: int
    n_equal: int
    similarity_pct: float
    n_missing_pred: int
    n_missing_gold: int
    n_missing_both: int
    n_missing_either: int
    macro_precision: Optional[float]
    macro_recall: Optional[float]
    macro_f1: Optional[float]
    balanced_accuracy: Optional[float]
    avg_jaccard: Optional[float]


@dataclass(frozen=True)
class EvalResult:
    n_joined_rows: int
    n_compared_total: int
    n_equal_total: int
    similarity_pct: float
    per_column_similarity_pct: Dict[str, float]
    columns: Dict[str, ColumnEvalResult]


# ============================================================
# Generic utils
# ============================================================

_WS_RE = re.compile(r"\s+")


def _safe_pct(num: float, den: float) -> float:
    return float("nan") if den == 0 else (100.0 * num / den)


def _safe_div(num: float, den: float) -> float:
    return float("nan") if den == 0 else (num / den)


def _is_missing(x: Any) -> bool:
    return pd.isna(x)


def _normalize_text(
    x: Any,
    *,
    casefold: bool = True,
    strip: bool = True,
    collapse_spaces: bool = True,
) -> Any:
    if _is_missing(x):
        return pd.NA
    s = str(x)
    if strip:
        s = s.strip()
    if collapse_spaces:
        s = _WS_RE.sub(" ", s)
    if casefold:
        s = s.upper()
    return s if s != "" else pd.NA


def _normalize_number(x: Any) -> Any:
    if _is_missing(x):
        return pd.NA
    try:
        v = float(x)
        if math.isnan(v):
            return pd.NA
        return v
    except Exception:
        return _normalize_text(x)


def _normalize_boolean(x: Any) -> Any:
    if _is_missing(x):
        return pd.NA
    if isinstance(x, bool):
        return x
    s = _normalize_text(x)
    if s is pd.NA:
        return pd.NA
    if s in {"TRUE", "T", "YES", "Y", "1"}:
        return True
    if s in {"FALSE", "F", "NO", "N", "0"}:
        return False
    return pd.NA


def _parse_list_value(x: Any, sep: str) -> List[str]:
    if _is_missing(x):
        return []

    if isinstance(x, list):
        vals = x
    else:
        s = str(x).strip()
        if s == "":
            return []

        # Try JSON list
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    vals = obj
                else:
                    vals = [s]
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                    vals = obj if isinstance(obj, list) else [s]
                except Exception:
                    vals = [v for v in s.split(sep)]
        else:
            vals = [v for v in s.split(sep)]

    out: List[str] = []
    for v in vals:
        nv = _normalize_text(v)
        if nv is not pd.NA:
            out.append(str(nv))
    return out


def normalize_series(s: pd.Series, cfg: ColumnConfig) -> pd.Series:
    if cfg.kind in {"label", "text"}:
        return s.map(
            lambda x: _normalize_text(
                x,
                casefold=cfg.casefold,
                strip=cfg.strip,
                collapse_spaces=cfg.collapse_spaces,
            )
        ).astype("object")

    if cfg.kind == "number":
        return s.map(_normalize_number).astype("object")

    if cfg.kind == "boolean":
        return s.map(_normalize_boolean).astype("object")

    if cfg.kind == "list":
        def _norm_list(x: Any) -> Any:
            vals = _parse_list_value(x, cfg.list_sep)
            if not vals:
                return pd.NA
            if cfg.list_as_set:
                vals = list(dict.fromkeys(vals))
            if cfg.sort_list_values:
                vals = sorted(vals)
            return tuple(vals)
        return s.map(_norm_list).astype("object")

    raise ValueError(f"Unsupported column kind: {cfg.kind}")


# ============================================================
# Metrics
# ============================================================

def _compute_classification_metrics(
    pred_s: pd.Series,
    gold_s: pd.Series,
) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[float], Optional[float]]:
    labels = sorted(set(pred_s.dropna().tolist()) | set(gold_s.dropna().tolist()))
    if not labels:
        empty = pd.DataFrame(columns=["label", "support", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1"])
        return empty, None, None, None, None

    rows = []
    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        tp = int(((pred_s == label) & (gold_s == label)).sum())
        fp = int(((pred_s == label) & (gold_s != label)).sum())
        fn = int(((pred_s != label) & (gold_s == label)).sum())
        support = int((gold_s == label).sum())
        pred_count = int((pred_s == label).sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = (
            float("nan")
            if (pd.isna(precision) or pd.isna(recall) or (precision + recall == 0))
            else 2 * precision * recall / (precision + recall)
        )

        rows.append(
            {
                "label": label,
                "support": support,
                "pred_count": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        if not pd.isna(precision):
            precisions.append(precision)
        if not pd.isna(recall):
            recalls.append(recall)
        if not pd.isna(f1):
            f1s.append(f1)

    df = pd.DataFrame(rows)
    macro_precision = None if not precisions else float(sum(precisions) / len(precisions))
    macro_recall = None if not recalls else float(sum(recalls) / len(recalls))
    macro_f1 = None if not f1s else float(sum(f1s) / len(f1s))
    balanced_accuracy = macro_recall
    return df, macro_precision, macro_recall, macro_f1, balanced_accuracy


def _compute_jaccard(pred_s: pd.Series, gold_s: pd.Series) -> float:
    scores: List[float] = []
    for p, g in zip(pred_s.tolist(), gold_s.tolist()):
        if _is_missing(p) or _is_missing(g):
            continue
        ps = set(p)
        gs = set(g)
        if not ps and not gs:
            continue
        scores.append(len(ps & gs) / len(ps | gs))
    return float("nan") if not scores else float(sum(scores) / len(scores))


# ============================================================
# Main compare
# ============================================================

def compare_frames(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    id_col: str,
    column_configs: List[ColumnConfig],
) -> Tuple[EvalResult, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Returns:
      - EvalResult
      - merged dataframe with normalized pred/gold columns + match flags
      - confusion matrices per column
      - classification reports per column
      - label distributions per column
    """
    if id_col not in pred.columns:
        raise ValueError(f"id_col '{id_col}' not found in pred")
    if id_col not in gold.columns:
        raise ValueError(f"id_col '{id_col}' not found in gold")

    cols = [cfg.name for cfg in column_configs]

    missing_pred = [c for c in cols if c not in pred.columns]
    missing_gold = [c for c in cols if c not in gold.columns]
    if missing_pred:
        raise ValueError(f"Missing columns in pred: {missing_pred}")
    if missing_gold:
        raise ValueError(f"Missing columns in gold: {missing_gold}")

    p = pred[[id_col] + cols].copy()
    g = gold[[id_col] + cols].copy()

    merged = p.merge(g, on=id_col, how="inner", suffixes=("_pred", "_gold"))
    n_joined_rows = len(merged)

    confusion_tables: Dict[str, pd.DataFrame] = {}
    classification_reports: Dict[str, pd.DataFrame] = {}
    label_distributions: Dict[str, pd.DataFrame] = {}
    col_results: Dict[str, ColumnEvalResult] = {}

    n_compared_total = 0
    n_equal_total = 0
    per_col_pct: Dict[str, float] = {}

    for cfg in column_configs:
        c = cfg.name

        pred_norm = normalize_series(merged[f"{c}_pred"], cfg)
        gold_norm = normalize_series(merged[f"{c}_gold"], cfg)

        merged[f"{c}__pred_norm"] = pred_norm
        merged[f"{c}__gold_norm"] = gold_norm

        pred_missing = pred_norm.isna()
        gold_missing = gold_norm.isna()
        both_missing = pred_missing & gold_missing
        either_missing = pred_missing | gold_missing

        n_missing_pred = int(pred_missing.sum())
        n_missing_gold = int(gold_missing.sum())
        n_missing_both = int(both_missing.sum())
        n_missing_either = int(either_missing.sum())

        if cfg.drop_na_pairs:
            ok = ~either_missing
            pred_eval = pred_norm[ok]
            gold_eval = gold_norm[ok]
        else:
            pred_eval = pred_norm.fillna("<NA>")
            gold_eval = gold_norm.fillna("<NA>")

        eq = (pred_eval == gold_eval)
        n_compared = int(len(pred_eval))
        n_equal = int(eq.sum())
        similarity_pct = _safe_pct(n_equal, n_compared)

        merged[f"{c}__compared"] = False
        merged.loc[pred_eval.index, f"{c}__compared"] = True
        merged[f"{c}__match"] = pd.NA
        merged.loc[pred_eval.index, f"{c}__match"] = eq.astype("boolean")

        macro_precision = None
        macro_recall = None
        macro_f1 = None
        balanced_accuracy = None
        avg_jaccard = None

        if cfg.kind in {"label", "text", "boolean", "number"}:
            cm = pd.crosstab(
                gold_eval.astype("string"),
                pred_eval.astype("string"),
                rownames=["gold"],
                colnames=["pred"],
                dropna=False,
            )
            confusion_tables[c] = cm.reset_index()

            cls_df, macro_precision, macro_recall, macro_f1, balanced_accuracy = _compute_classification_metrics(
                pred_eval, gold_eval
            )
            classification_reports[c] = cls_df

            dist = pd.DataFrame(
                {
                    "label": sorted(set(pred_eval.dropna().tolist()) | set(gold_eval.dropna().tolist()))
                }
            )
            if not dist.empty:
                dist["gold_count"] = dist["label"].map(gold_eval.value_counts(dropna=False)).fillna(0).astype(int)
                dist["pred_count"] = dist["label"].map(pred_eval.value_counts(dropna=False)).fillna(0).astype(int)
                dist["gold_pct"] = dist["gold_count"].map(lambda x: _safe_pct(x, len(gold_eval)))
                dist["pred_pct"] = dist["pred_count"].map(lambda x: _safe_pct(x, len(pred_eval)))
            label_distributions[c] = dist

        elif cfg.kind == "list":
            pair_counts = (
                pd.DataFrame(
                    {
                        "gold": gold_eval.map(lambda x: json.dumps(list(x), ensure_ascii=False)),
                        "pred": pred_eval.map(lambda x: json.dumps(list(x), ensure_ascii=False)),
                    }
                )
                .value_counts()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .reset_index(drop=True)
            )
            confusion_tables[c] = pair_counts
            classification_reports[c] = pd.DataFrame()
            label_distributions[c] = pd.DataFrame()
            avg_jaccard = _compute_jaccard(pred_eval, gold_eval)

        else:
            confusion_tables[c] = pd.DataFrame()
            classification_reports[c] = pd.DataFrame()
            label_distributions[c] = pd.DataFrame()

        col_results[c] = ColumnEvalResult(
            column=c,
            kind=cfg.kind,
            n_joined_rows=n_joined_rows,
            n_compared=n_compared,
            n_equal=n_equal,
            similarity_pct=similarity_pct,
            n_missing_pred=n_missing_pred,
            n_missing_gold=n_missing_gold,
            n_missing_both=n_missing_both,
            n_missing_either=n_missing_either,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            balanced_accuracy=balanced_accuracy,
            avg_jaccard=avg_jaccard,
        )

        per_col_pct[c] = similarity_pct
        n_compared_total += n_compared
        n_equal_total += n_equal

    result = EvalResult(
        n_joined_rows=n_joined_rows,
        n_compared_total=n_compared_total,
        n_equal_total=n_equal_total,
        similarity_pct=_safe_pct(n_equal_total, n_compared_total),
        per_column_similarity_pct=per_col_pct,
        columns=col_results,
    )

    return result, merged, confusion_tables, classification_reports, label_distributions


# ============================================================
# Export helpers
# ============================================================

def save_eval_outputs(
    out_dir: str | Path,
    result: EvalResult,
    merged: pd.DataFrame,
    confusion_tables: Dict[str, pd.DataFrame],
    classification_reports: Dict[str, pd.DataFrame],
    label_distributions: Dict[str, pd.DataFrame],
    column_configs: List[ColumnConfig],
    id_col: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_joined_rows": result.n_joined_rows,
        "n_compared_total": result.n_compared_total,
        "n_equal_total": result.n_equal_total,
        "similarity_pct": result.similarity_pct,
        "per_column_similarity_pct": result.per_column_similarity_pct,
        "columns": {k: asdict(v) for k, v in result.columns.items()},
    }

    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_rows = []
    for c, r in result.columns.items():
        summary_rows.append(asdict(r))
    pd.DataFrame(summary_rows).to_csv(out_dir / "eval_summary.csv", index=False)

    for cfg in column_configs:
        c = cfg.name

        if c in confusion_tables and not confusion_tables[c].empty:
            confusion_tables[c].to_csv(out_dir / f"confusion_{c}.csv", index=False)

        if c in classification_reports and not classification_reports[c].empty:
            classification_reports[c].to_csv(out_dir / f"classification_report_{c}.csv", index=False)

        if c in label_distributions and not label_distributions[c].empty:
            label_distributions[c].to_csv(out_dir / f"label_distribution_{c}.csv", index=False)

        pred_col = f"{c}__pred_norm"
        gold_col = f"{c}__gold_norm"
        match_col = f"{c}__match"
        compared_col = f"{c}__compared"

        error_cols = [id_col, pred_col, gold_col, match_col, compared_col]
        error_df = merged.loc[
            merged[compared_col].fillna(False) & (merged[match_col] == False),
            error_cols,
        ].copy()

        rename_map = {
            pred_col: f"{c}_pred_norm",
            gold_col: f"{c}_gold_norm",
            match_col: "is_match",
            compared_col: "was_compared",
        }
        error_df = error_df.rename(columns=rename_map)
        error_df.to_csv(out_dir / f"errors_{c}.csv", index=False)
