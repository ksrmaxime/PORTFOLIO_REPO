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


# -------------------------
# Robust JSON loading helpers (run3-only patch)
# -------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_object(s: str) -> str | None:
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else None


def _repair_common_json_prefix_bugs(s: str) -> str:
    """
    Conservative repairs for common model artifacts we saw in logs:
    - leading comma before first key: , "justification": ...
    - missing opening brace: "justification": ...
    """
    t = s.lstrip()

    # Case 1: starts with comma then a key
    if t.startswith(","):
        t2 = t.lstrip(", \n\r\t")
        if t2.startswith('"justification"') or t2.startswith('"targets"') or t2.startswith('"instruments"'):
            return "{" + t2 + "}"

    # Case 2: starts with a key without opening brace
    if t.startswith('"justification"') or t.startswith('"targets"') or t.startswith('"instruments"'):
        return "{" + t + "}"

    return s


def load_json_robust(raw: str):
    if raw is None:
        return None

    s = _strip_code_fences(str(raw))
    s = _repair_common_json_prefix_bugs(s)

    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try extracting first {...} block
    block = _extract_first_json_object(s)
    if block:
        try:
            return json.loads(block)
        except Exception:
            return None

    return None


# -------------------------
# Parsing logic (keeps original "justification/targets/instruments" outputs)
# -------------------------
ALLOWED_TARGETS = {"Infrastructure", "Data", "Skills", "Adoption"}
ALLOWED_INSTRUMENTS = {
    "Voluntary instruments",
    "Taxes & Subsidies",
    "Public Investment & Public procurement",
    "Prohibition & Ban",
    "Planning & evaluation instruments",
    "Obligation",
    "Liability scheme",
}


def parse_run3_json(raw: str):
    """
    Expected strict-ish JSON:
    {
      "justification": "...",
      "targets": [...],
      "instruments": [...]
    }

    Returns (justif, targets_list, instruments_list) or (None,None,None) on failure.
    """
    obj = load_json_robust(raw)
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
    targets = [t.strip() for t in targets if isinstance(t, str) and t.strip()]
    instruments = [i.strip() for i in instruments if isinstance(i, str) and i.strip()]

    # Keep only allowed labels (do not hard-fail; drop invalids)
    targets = [t for t in targets if t in ALLOWED_TARGETS]
    instruments = [i for i in instruments if i in ALLOWED_INSTRUMENTS]

    # Still require at least one of each for a "successful" parse
    if not targets or not instruments:
        return None, None, None

    return justif, targets, instruments


def parse_output(raw: str, targets_col: str, instruments_col: str, justif_col: str) -> dict:
    justif, targets, instruments = parse_run3_json(raw)

    if justif is None:
    head = "" if raw is None else str(raw)[:600].replace("\n","\\n")
    print(f"[RUN3 PARSE FAIL] raw_head={head}")
    return {targets_col: pd.NA, instruments_col: pd.NA, justif_col: pd.NA}

    #if justif is None:
        # keep NA to allow reruns / diagnostics
      #  return {targets_col: pd.NA, instruments_col: pd.NA, justif_col: pd.NA}

    return {
        targets_col: "; ".join(targets),
        instruments_col: "; ".join(instruments),
        justif_col: justif,
    }


# -------------------------
# Main
# -------------------------
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
    ap.add_argument("--max_new_tokens", type=int, default=400)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    send_mask = build_run3_mask(
        df,
        level_col=args.level_col,
        relevant_art_col=args.relevant_art_col,
    )

    # ensure output columns exist (string dtype) — created in run3, so cannot be "already filled" on first run
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
        skip_if_already_filled=args.justif_col,
    )

    job_id = os.environ.get("SLURM_JOB_ID") or args.job_id or "nojobid"
    base = f"{args.output_base}_job{job_id}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"Saved: {parquet_path} and {csv_path}")
    print(f"Selected rows (mask True): {int(send_mask.sum()):,} / {len(df):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())