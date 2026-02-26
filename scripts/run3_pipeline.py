# scripts/run3_pipeline.py
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

from src.run3_config import build_run3_mask
from src import run3_prompts


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # remove ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_run3_json(raw: str):
    """
    Expected strict JSON:
    {
      "justification": "...",
      "targets": [...],
      "instruments": [...]
    }
    Returns (justif, targets_list, instruments_list) or (None,None,None) on failure.
    """
    if raw is None:
        return None, None, None

    s = _strip_code_fences(str(raw))

    # try direct JSON
    try:
        obj = json.loads(s)
    except Exception:
        # fallback: attempt to extract first {...} block
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None, None, None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None, None, None

    if not isinstance(obj, dict):
        return None, None, None

    justif = obj.get("justification")
    targets = obj.get("targets")
    instruments = obj.get("instruments")

    if not isinstance(justif, str) or not justif.strip():
        return None, None, None
    if not isinstance(targets, list) or not all(isinstance(x, str) for x in targets):
        return None, None, None
    if not isinstance(instruments, list) or not all(isinstance(x, str) for x in instruments):
        return None, None, None

    justif = " ".join(justif.strip().split())
    targets = [t.strip() for t in targets if t.strip()]
    instruments = [i.strip() for i in instruments if i.strip()]

    # Optional: enforce allowed labels (drop invalid instead of failing hard)
    allowed_targets = {"Data", "Computing Infrastructure", "Development & Adoption", "Skills"}
    allowed_instruments = {
        "Voluntary instrument",
        "Tax/Subsidy",
        "Public investment & procurement",
        "Prohibition/Ban",
        "Planning & experimentation",
        "Obligation",
        "Liability scheme",
    }

    targets = [t for t in targets if t in allowed_targets]
    instruments = [i for i in instruments if i in allowed_instruments]

    return justif, targets, instruments

def parse_output(raw: str, targets_col: str, instruments_col: str, justif_col: str) -> dict:
    justif, targets, instruments = parse_run3_json(raw)

    if justif is None:
        head = "" if raw is None else str(raw)[:600].replace("\n", "\\n")
        print(f"[RUN3 PARSE FAIL] raw_head={head}")
        return {targets_col: pd.NA, instruments_col: pd.NA, justif_col: pd.NA}

    # ... le reste inchangé ...
#def parse_output(raw: str, targets_col: str, instruments_col: str, justif_col: str) -> dict:
    #justif, targets, instruments = parse_run3_json(raw)

    # If parsing fails: leave NA (so reruns can pick it up)
    #if justif is None:
     #   return {targets_col: pd.NA, instruments_col: pd.NA, justif_col: pd.NA}

    targets_s = "; ".join(targets) if targets else ""
    instruments_s = "; ".join(instruments) if instruments else ""

    return {
        targets_col: targets_s,
        instruments_col: instruments_s,
        justif_col: justif,
    }


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
    ap.add_argument("--relevant_art_col", default="RELEVANT_ART")

    ap.add_argument("--targets_col", default="TARGETS")
    ap.add_argument("--instruments_col", default="INSTRUMENTS")
    ap.add_argument("--justif_col", default="RUN3_JUSTIF")

    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=220)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    send_mask = build_run3_mask(
        df,
        level_col=args.level_col,
        relevant_art_col=args.relevant_art_col,
    )

    # create/normalize columns
    for c in (args.targets_col, args.instruments_col, args.justif_col):
        if c not in df.columns:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            df[c] = df[c].astype("string")

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
        return run3_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        return parse_output(raw, args.targets_col, args.instruments_col, args.justif_col)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run3_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[args.targets_col, args.instruments_col, args.justif_col],
        skip_if_already_filled=args.justif_col,  # reprise sur la justification
    )

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