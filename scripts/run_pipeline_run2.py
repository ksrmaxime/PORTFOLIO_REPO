import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import re
from pathlib import Path
import pandas as pd

from src.client import TransformersClient, LLMConfig
from src.runner import run_llm_dataframe, RunConfig
import src.prompts as prompts

from src.relevant_art_selection import build_articles_to_send_mask


def parse_relevant_justif(raw: str):
    if raw is None:
        return None, None
    s = str(raw)

    rel_matches = re.findall(r"RELEVANT:\s*(TRUE|FALSE)\b", s, flags=re.IGNORECASE)
    jus_matches = re.findall(r"JUSTIFICATION:\s*(.+)", s, flags=re.IGNORECASE)

    if not rel_matches or not jus_matches:
        return None, None

    rel_token = rel_matches[-1].upper()
    jus = jus_matches[-1].strip()

    if rel_token not in ("TRUE", "FALSE") or not jus:
        return None, None

    relevant = (rel_token == "TRUE")
    jus_clean = " ".join(jus.split())
    return relevant, jus_clean


def parse_output(raw: str, decision_col: str, justif_col: str) -> dict:
    decision, justif = parse_relevant_justif(raw)

    # Si parsing échoue: on laisse NA -> tu pourras relancer plus tard, ça sera re-pické automatiquement
    if decision is None or justif is None:
        return {decision_col: pd.NA, justif_col: pd.NA}

    return {decision_col: bool(decision), justif_col: justif}


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="text")
    ap.add_argument("--level_col", default="level")
    ap.add_argument("--relevance_col", default=None)

    # ✅ فقط 2 colonnes
    ap.add_argument("--decision_col", default="RELEVANT_ART")
    ap.add_argument("--justif_col", default="RELEVANT_ART_JUSTIF")

    ap.add_argument("--batch_size", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=160)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    send_mask = build_articles_to_send_mask(
        df,
        level_col=args.level_col,
        relevance_col=args.relevance_col,
    )

    # ✅ crée uniquement 2 colonnes
    if args.decision_col not in df.columns:
        df[args.decision_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    else:
        df[args.decision_col] = df[args.decision_col].astype("boolean")

    if args.justif_col not in df.columns:
        df[args.justif_col] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df[args.justif_col] = df[args.justif_col].astype("string")

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
        return prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        return parse_output(raw, args.decision_col, args.justif_col)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[args.decision_col, args.justif_col],
        skip_if_already_filled=args.decision_col,  # ✅ reprise sur décision seulement
    )

    # --- WRITE (parquet + csv, suffix job id) ---
    job_id = os.environ.get("SLURM_JOB_ID") or args.job_id or "nojobid"

    base = f"{args.output_base}_job{job_id}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"Saved: {parquet_path} and {csv_path} | Selected: {int(send_mask.sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())