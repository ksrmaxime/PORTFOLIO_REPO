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
from src import run_target_control_prompts
from src.run_target_prompts import TARGET_DEFINITIONS


def parse_output(raw: str, decision_col: str, justif_col: str) -> dict:
    if raw is None:
        return {decision_col: pd.NA, justif_col: pd.NA}

    text = str(raw).strip()

    justif = pd.NA
    m_justif = re.search(r"Justification\s*:\s*(.+?)(?=Décision\s*:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if m_justif:
        justif = " ".join(m_justif.group(1).strip().split())

    decision = pd.NA
    m_dec = re.search(r"Décision\s*:\s*(OUI|NON|YES|NO)\b", text, flags=re.IGNORECASE)
    if m_dec:
        decision = m_dec.group(1).upper() in {"OUI", "YES"}
    else:
        m_bare = re.search(r"\b(OUI|NON|YES|NO)\b", text, flags=re.IGNORECASE)
        if m_bare:
            decision = m_bare.group(1).upper() in {"OUI", "YES"}

    if decision is pd.NA:
        head = text[:300].replace("\n", "\\n")
        print(f"[RUN_TARGET_CONTROL PARSE FAIL] raw_head={head}")

    return {decision_col: bool(decision) if decision is not pd.NA else pd.NA, justif_col: justif}


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--target_code", required=True, choices=list(TARGET_DEFINITIONS))

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="text")

    ap.add_argument("--decision_col", default=None)
    ap.add_argument("--justif_col", default=None)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=150)

    args = ap.parse_args()

    source_col = f"target_{args.target_code}"
    decision_col = args.decision_col or f"control_target_{args.target_code}"
    justif_col = args.justif_col or f"control_target_{args.target_code}_JUSTIF"

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    if source_col not in df.columns:
        raise KeyError(
            f"Colonne '{source_col}' absente de l'input — le run de contrôle doit être "
            f"alimenté avec la sortie de la chaîne run_target (qui produit cette colonne)."
        )

    # On ne reprend que les articles classés OUI au premier passage.
    send_mask = df[source_col].fillna(False).astype(bool)

    df[decision_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df[justif_col] = pd.Series(pd.NA, index=df.index, dtype="string")

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
        return run_target_control_prompts.build_user_prompt(row, text_col=text_col, code=args.target_code)

    def _parse(raw: str) -> dict:
        return parse_output(raw, decision_col, justif_col)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run_target_control_prompts.build_system_prompt(args.target_code),
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[decision_col, justif_col],
        skip_if_already_filled=decision_col,
        required_cols=[decision_col],
        max_retries=2,
    )

    job_id = os.environ.get("SLURM_JOB_ID") or args.job_id or "nojobid"

    base = f"{args.output_base}_job{job_id}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)

    control_cols = [c for c in [decision_col, justif_col] if c in out.columns]
    base_cols = [c for c in out.columns if c not in control_cols]
    out = out[base_cols + control_cols]

    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    n_sent = int(send_mask.sum())
    n_confirmed = int(out[decision_col].eq(True).sum())
    n_infirmed = int(out[decision_col].eq(False).sum())
    n_na = int(out.loc[send_mask, decision_col].isna().sum())
    print(f"Saved: {parquet_path} and {csv_path}")
    print(
        f"{decision_col} — envoyés: {n_sent:,} | confirmés OUI: {n_confirmed:,} | "
        f"infirmés NON: {n_infirmed:,} | NA (parse fail): {n_na:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
