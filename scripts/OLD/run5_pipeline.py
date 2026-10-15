import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import argparse
import json
import re
import pandas as pd

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src import run5_prompts
from src.run5_config import build_run5_mask

TARGET_CODES = run5_prompts.TARGET_CODES
INSTRUMENT_CODES = run5_prompts.INSTRUMENT_CODES
MAX_ENTRIES = run5_prompts.MAX_ENTRIES

_ENTRY_RE = re.compile(
    r"Cible\s*:\s*([A-Za-z_]+).*?"
    r"Instrument\s*:\s*([A-Za-z_]+).*?"
    r"Justification\s*:\s*(.+?)(?=(?:\n\s*Cible\s*:)|(?:\n\s*-{3,}\s*$)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_output(raw: str) -> dict:
    cols = {
        "RUN5_RAW": pd.NA,
        "N_PORTFOLIO_ENTRIES": 0,
        "PORTFOLIO_ENTRIES_JSON": pd.NA,
        "TARGET_CODES": pd.NA,
        "INSTRUMENT_CODES": pd.NA,
    }

    if raw is None:
        return cols

    text = str(raw).strip()
    cols["RUN5_RAW"] = text

    entries: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Normalize entry separators so the regex only has to deal with one shape.
    blocks = re.split(r"\n\s*-{3,}\s*\n", text)

    for block in blocks:
        m = _ENTRY_RE.search(block)
        if not m:
            continue

        target_code = m.group(1).strip().upper()
        instrument_code = m.group(2).strip().upper()
        justif = " ".join(m.group(3).strip().split())

        if target_code not in TARGET_CODES or instrument_code not in INSTRUMENT_CODES:
            continue

        pair = (target_code, instrument_code)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        entries.append(
            {
                "target_code": target_code,
                "target_label": TARGET_CODES[target_code],
                "instrument_code": instrument_code,
                "instrument_label": INSTRUMENT_CODES[instrument_code],
                "justification": justif,
            }
        )

        if len(entries) >= MAX_ENTRIES:
            break

    if not entries:
        head = text[:300].replace("\n", "\\n")
        print(f"[RUN5 PARSE FAIL] raw_head={head}")
        return cols

    cols["N_PORTFOLIO_ENTRIES"] = len(entries)
    cols["PORTFOLIO_ENTRIES_JSON"] = json.dumps(entries, ensure_ascii=False)
    cols["TARGET_CODES"] = "|".join(sorted({e["target_code"] for e in entries}))
    cols["INSTRUMENT_CODES"] = "|".join(sorted({e["instrument_code"] for e in entries}))
    return cols


def build_entries_frame(df: pd.DataFrame, id_col: str, carry_cols: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        raw_json = row.get("PORTFOLIO_ENTRIES_JSON")
        if pd.isna(raw_json):
            continue
        try:
            entries = json.loads(raw_json)
        except Exception:
            continue

        base = {c: row.get(c) for c in carry_cols if c in df.columns}
        for e in entries:
            rows.append({id_col: row[id_col], **base, **e})

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="text")
    ap.add_argument("--final_ai_relevant_col", default="final_ai_relevant")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=400)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    send_mask = build_run5_mask(df, final_ai_relevant_col=args.final_ai_relevant_col)

    print(f"Rows total: {len(df):,} | {args.final_ai_relevant_col} == True (sent to run5): {int(send_mask.sum()):,}")

    output_cols = [
        "RUN5_RAW",
        "N_PORTFOLIO_ENTRIES",
        "PORTFOLIO_ENTRIES_JSON",
        "TARGET_CODES",
        "INSTRUMENT_CODES",
    ]
    for c in output_cols:
        df[c] = pd.Series(pd.NA, index=df.index, dtype="object")

    client = TransformersClient(
        LLMConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )

    run_cfg = RunConfig(
        id_col="row_uid" if "row_uid" in df.columns else "__index__",
        text_col=args.text_col,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    def _select_mask(df_: pd.DataFrame) -> pd.Series:
        return send_mask

    def _build_prompt(row: pd.Series, text_col: str) -> str:
        return run5_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        return parse_output(raw)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run5_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=output_cols,
        skip_if_already_filled="RUN5_RAW",
    )

    job_id = os.environ.get("SLURM_JOB_ID") or args.job_id or "nojobid"

    base = f"{args.output_base}_job{job_id}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)

    portfolio_cols = [c for c in output_cols if c in out.columns]
    base_cols = [c for c in out.columns if c not in portfolio_cols]
    out = out[base_cols + portfolio_cols]

    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    # Long-format portfolio dataset: one row per (article, target, instrument) entry —
    # this is the structured output described in section 3.2 of the PoC (Stage 2).
    carry_cols = [c for c in ["law_id", "label", "ctx_law_title", "ctx_chapter_title", "ctx_article_title", args.text_col] if c in out.columns]
    entries_df = build_entries_frame(out.loc[send_mask], id_col="row_id", carry_cols=carry_cols)

    entries_parquet_path = f"{args.output_base}_entries_job{job_id}.parquet"
    entries_csv_path = f"{args.output_base}_entries_job{job_id}.csv"
    entries_df.to_parquet(entries_parquet_path, index=False)
    entries_df.to_csv(entries_csv_path, index=False)

    n_sent = int(send_mask.sum())
    n_parsed = int(out.loc[send_mask, "N_PORTFOLIO_ENTRIES"].fillna(0).astype(int).gt(0).sum())
    n_na = n_sent - n_parsed
    n_entries = len(entries_df)
    print(f"Saved: {parquet_path} and {csv_path}")
    print(f"Saved portfolio entries: {entries_parquet_path} and {entries_csv_path}")
    print(f"Articles sent: {n_sent:,} | coded (>=1 entry): {n_parsed:,} | parse fail: {n_na:,} | total entries: {n_entries:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
