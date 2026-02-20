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

# LLM config
DTYPE = "bf16"          # "bf16" or "fp16"
TEMPERATURE = 0.0
MAX_TOKENS = 32         # max_new_tokens

# Batching
BATCH_SIZE = 40         # 32–64 typically ok on A100 40GB for short prompts

# Safety checkpoint
CHECKPOINT_EVERY_BATCHES = 25  # save every 25 batches
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

    df = pd.read_parquet(in_path)

    if TEXT_COL not in df.columns:
        raise KeyError(f"Missing column: {TEXT_COL}")
    if LEVEL_COL not in df.columns:
        raise KeyError(f"Missing column: {LEVEL_COL}")

    send_mask = build_articles_to_send_mask(df, level_col=LEVEL_COL, relevance_col=RELEVANCE_COL)

    # output column: nullable boolean, only filled for level==6 we actually send
    if NEW_COL not in df.columns:
        df[NEW_COL] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    else:
        df[NEW_COL] = df[NEW_COL].astype("boolean")

    # only send those still NA (resume-friendly)
    todo_idx = df.index[send_mask & df[NEW_COL].isna()]

    print(f"[INFO] Total rows: {len(df):,}")
    print(f"[INFO] Selected level==6 to send: {int(send_mask.sum()):,}")
    print(f"[INFO] Remaining to classify (NA among selected): {len(todo_idx):,}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")

    clf = ApertusBatchClassifier(
        ApertusConfig(
            model_path=MODEL_PATH,
            dtype=DTYPE,
            trust_remote_code=True,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    )

    batches_done = 0

    for start in tqdm(range(0, len(todo_idx), BATCH_SIZE), desc="Batches", unit="batch"):
        batch_idx = todo_idx[start : start + BATCH_SIZE]

        batch_texts = []
        for idx in batch_idx:
            t = df.at[idx, TEXT_COL]
            t = "" if pd.isna(t) else str(t)
            batch_texts.append(t)

        preds = clf.classify_batch(batch_texts)  # list[Optional[bool]]

        # write back
        for idx, pred, txt in zip(batch_idx, preds, batch_texts):
            if not txt.strip():
                df.at[idx, NEW_COL] = False
            else:
                df.at[idx, NEW_COL] = pred if pred is not None else pd.NA

        batches_done += 1
        if batches_done % CHECKPOINT_EVERY_BATCHES == 0:
            df.to_parquet(ckpt_parquet, index=False)

    # final writes
    df.to_parquet(out_parquet, index=False)

    if WRITE_CSV:
        df_csv = df.copy()
        df_csv[NEW_COL] = df_csv[NEW_COL].map(_to_csv_friendly_bool)
        df_csv.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[OK] Wrote parquet: {out_parquet}")
    if WRITE_CSV:
        print(f"[OK] Wrote csv:    {out_csv}")
    if ckpt_parquet.exists():
        print(f"[INFO] Checkpoint: {ckpt_parquet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())