from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from portfolio_repo.llm.relevant_art_selection import build_articles_to_send_mask
from portfolio_repo.llm.relevant_art_apertus_runner import ApertusBatchClassifier, ApertusConfig


# =========================
# EDIT THESE (INPUT/OUTPUT)
# =========================
INPUT_PARQUET = "/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure_selected_with_ai_relevant.parquet"
OUTPUT_DIR = "/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed"
# =========================

MODEL_PATH = "/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509"

TEXT_COL = "text"
LEVEL_COL = "level"
RELEVANCE_COL = None  # auto-detect RELEVANT_AI or AI_RELEVANT

NEW_COL = "RELEVANT_ART"
RAW_COL = "RELEVANT_ART_RAW"
PARSEOK_COL = "RELEVANT_ART_PARSE_OK"

# LLM config
DTYPE = "bf16"
TEMPERATURE = 0.0
MAX_TOKENS = 16  # plus bas suffit, réduit le blabla

# Batching
BATCH_SIZE = 40

# Safety checkpoint
CHECKPOINT_EVERY_BATCHES = 25
WRITE_CSV = True


def _to_csv_friendly_bool(x):
    return "" if pd.isna(x) else bool(x)


def main() -> int:
    in_path = Path(INPUT_PARQUET)
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
    stem = in_path.stem

    out_parquet = out_dir / f"{stem}_with_{NEW_COL.lower()}_job{job_id}.parquet"
    out_csv = out_dir / f"{stem}_with_{NEW_COL.lower()}_job{job_id}.csv"
    ckpt_parquet = out_dir / f"{stem}_with_{NEW_COL.lower()}_job{job_id}.checkpoint.parquet"

    debug_dir = out_dir / f"run2_debug_job{job_id}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    parse_fail_csv = debug_dir / "parse_failures.csv"

    df = pd.read_parquet(in_path)

    for c in (TEXT_COL, LEVEL_COL):
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    send_mask = build_articles_to_send_mask(df, level_col=LEVEL_COL, relevance_col=RELEVANCE_COL)

    # Ensure columns exist with correct dtypes
    if NEW_COL not in df.columns:
        df[NEW_COL] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    else:
        df[NEW_COL] = df[NEW_COL].astype("boolean")

    if RAW_COL not in df.columns:
        df[RAW_COL] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df[RAW_COL] = df[RAW_COL].astype("string")

    if PARSEOK_COL not in df.columns:
        df[PARSEOK_COL] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    else:
        df[PARSEOK_COL] = df[PARSEOK_COL].astype("boolean")

    # Resume-friendly: only rows selected and NEW_COL is NA
    todo_idx = df.index[send_mask & df[NEW_COL].isna()]

    print(f"[INFO] Total rows: {len(df):,}")
    print(f"[INFO] Selected level==6 to send: {int(send_mask.sum()):,}")
    print(f"[INFO] Remaining to classify (NA among selected): {len(todo_idx):,}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Debug dir: {debug_dir}")

    clf = ApertusBatchClassifier(
        ApertusConfig(
            model_path=MODEL_PATH,
            dtype=DTYPE,
            trust_remote_code=True,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    )

    failures = []  # collect parse failures rows (small)
    batches_done = 0

    for start in tqdm(range(0, len(todo_idx), BATCH_SIZE), desc="Batches", unit="batch"):
        batch_idx = todo_idx[start : start + BATCH_SIZE]

        batch_texts = []
        for idx in batch_idx:
            t = df.at[idx, TEXT_COL]
            t = "" if pd.isna(t) else str(t)
            batch_texts.append(t)

        preds, raws = clf.classify_batch_raw(batch_texts)

        for idx, pred, raw, txt in zip(batch_idx, preds, raws, batch_texts):
            if not txt.strip():
                # deterministic: empty text => False
                df.at[idx, NEW_COL] = False
                df.at[idx, PARSEOK_COL] = True
                df.at[idx, RAW_COL] = ""
                continue

            df.at[idx, RAW_COL] = raw

            if pred is None:
                # strict parse failed => leave NA so we can debug + rerun
                df.at[idx, NEW_COL] = pd.NA
                df.at[idx, PARSEOK_COL] = False
                failures.append(
                    {
                        "row_index": int(idx) if isinstance(idx, (int,)) else str(idx),
                        "raw": raw,
                        "text_preview": (txt[:300] + "…") if len(txt) > 300 else txt,
                    }
                )
            else:
                df.at[idx, NEW_COL] = pred
                df.at[idx, PARSEOK_COL] = True

        batches_done += 1
        if batches_done % CHECKPOINT_EVERY_BATCHES == 0:
            df.to_parquet(ckpt_parquet, index=False)
            if failures:
                pd.DataFrame(failures).to_csv(parse_fail_csv, index=False, encoding="utf-8")

    # Write final outputs
    df.to_parquet(out_parquet, index=False)

    if WRITE_CSV:
        df_csv = df.copy()
        df_csv[NEW_COL] = df_csv[NEW_COL].map(_to_csv_friendly_bool)
        df_csv[PARSEOK_COL] = df_csv[PARSEOK_COL].map(_to_csv_friendly_bool)
        df_csv.to_csv(out_csv, index=False, encoding="utf-8")

    if failures:
        pd.DataFrame(failures).to_csv(parse_fail_csv, index=False, encoding="utf-8")
        print(f"[WARN] Parse failures: {len(failures):,} -> {parse_fail_csv}")
    else:
        print("[OK] No parse failures (strict TRUE/FALSE).")

    print(f"[OK] Wrote parquet: {out_parquet}")
    if WRITE_CSV:
        print(f"[OK] Wrote csv:    {out_csv}")
    if ckpt_parquet.exists():
        print(f"[INFO] Checkpoint: {ckpt_parquet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())