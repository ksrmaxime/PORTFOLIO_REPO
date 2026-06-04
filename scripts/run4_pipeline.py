import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import argparse
import re
import pandas as pd

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
from src import run4_prompts
from src.run4_config import build_run4_mask


def parse_output(raw: str, decision_col: str, audit_col: str) -> dict:
    if raw is None:
        return {decision_col: pd.NA, audit_col: pd.NA}

    text = str(raw).strip()

    audit = pd.NA
    m_audit = re.search(r"Audit\s*:\s*(.+?)(?=Décision\s*:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if m_audit:
        audit = " ".join(m_audit.group(1).strip().split())

    decision = pd.NA
    m_dec = re.search(r"Décision\s*:\s*(CONFIRME|INFIRME)\b", text, flags=re.IGNORECASE)
    if m_dec:
        decision = m_dec.group(1).upper() == "CONFIRME"
    else:
        m_bare = re.search(r"\b(CONFIRME|INFIRME)\b", text, flags=re.IGNORECASE)
        if m_bare:
            decision = m_bare.group(1).upper() == "CONFIRME"

    if decision is pd.NA:
        head = text[:300].replace("\n", "\\n")
        print(f"[RUN4 PARSE FAIL] raw_head={head}")

    return {decision_col: bool(decision) if decision is not pd.NA else pd.NA, audit_col: audit}


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="text")
    ap.add_argument("--ai_relevant_col", default="AI_RELEVANT")
    ap.add_argument("--justif_col", default="RUN3_JUSTIF")

    ap.add_argument("--decision_col", default="AI_CONFIRMED")
    ap.add_argument("--audit_col", default="RUN4_AUDIT")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=150)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    send_mask = build_run4_mask(df, ai_relevant_col=args.ai_relevant_col)

    print(f"Rows total: {len(df):,} | AI_RELEVANT classified (True+False): {int(send_mask.sum()):,}")

    df[args.decision_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df[args.audit_col] = pd.Series(pd.NA, index=df.index, dtype="string")

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
        return run4_prompts.build_user_prompt(row, text_col=text_col, justif_col=args.justif_col)

    def _parse(raw: str) -> dict:
        return parse_output(raw, args.decision_col, args.audit_col)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run4_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[args.decision_col, args.audit_col],
        skip_if_already_filled=args.audit_col,
    )

    job_id = os.environ.get("SLURM_JOB_ID") or args.job_id or "nojobid"

    base = f"{args.output_base}_job{job_id}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)

    ai_cols = [c for c in [args.decision_col, args.audit_col] if c in out.columns]
    base_cols = [c for c in out.columns if c not in ai_cols]
    out = out[base_cols + ai_cols]

    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    n_confirme = int(out[args.decision_col].eq(True).sum())
    n_infirme = int(out[args.decision_col].eq(False).sum())
    n_na = int(out[args.decision_col].isna().sum())
    print(f"Saved: {parquet_path} and {csv_path}")
    print(f"AI_CONFIRMED — CONFIRME: {n_confirme:,} | INFIRME: {n_infirme:,} | NA (parse fail): {n_na:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
