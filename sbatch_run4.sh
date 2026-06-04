#!/bin/bash -l
#SBATCH --job-name=run4_ai_confirmed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run4_ai_confirmed_%j.out
#SBATCH --error=logs/run4_ai_confirmed_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed
OUTBASE="/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure_with_ai_confirmed"

# Input = output du run 3 (remplacer JOB_ID par le job ID du run 3)
RUN3_JOB_ID=
INPUT="${OUTDIR}/laws_structure_with_ai_relevant_job${RUN3_JOB_ID}.parquet"

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

python scripts/run4_pipeline.py \
  --input  "$INPUT" \
  --output_base "$OUTBASE" \
  --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --trust_remote_code \
  --text_col text \
  --ai_relevant_col AI_RELEVANT \
  --justif_col RUN3_JUSTIF \
  --decision_col AI_CONFIRMED \
  --audit_col RUN4_AUDIT \
  --batch_size 8 \
  --max_new_tokens 150 \
  --temperature 0.0

# --- Évaluation ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"

GOLD_CSV="data/external/PORTFOLIO_GOLD.csv"

TEMP_RUN_DIR="data/output/run4_job${SLURM_JOB_ID}"
mkdir -p "$TEMP_RUN_DIR"

GOLD_WITH_ID="${TEMP_RUN_DIR}/gold_with_row_id.csv"
python scripts/add_row_id.py "$GOLD_CSV" --col row_id --overwrite --out "$GOLD_WITH_ID"

# Renommer AI_CONFIRMED → AI_RELEVANT dans une copie temporaire du pred pour le scoring
PRED_FOR_SCORE="${TEMP_RUN_DIR}/pred_for_scoring.csv"
python -c "
import pandas as pd
df = pd.read_csv('$PRED_CSV')
df = df.rename(columns={'AI_CONFIRMED': 'AI_RELEVANT'})
df.to_csv('$PRED_FOR_SCORE', index=False)
"

SCORE_LOG=$(python scripts/score.py \
  --pred "$PRED_FOR_SCORE" \
  --gold "$GOLD_WITH_ID" \
  --id_col row_id \
  --cols AI_RELEVANT \
  --col_kinds AI_RELEVANT=label \
  --rename_gold_cols Instrument=instrument,AI_Relevant=AI_RELEVANT \
  --extra_cols text \
  --report_dir "$TEMP_RUN_DIR/eval")

echo "$SCORE_LOG"

SCORE=$(echo "$SCORE_LOG" | awk '/^Similarity:/ {gsub(/%/,"",$2); print $2; exit}')
SCORE=${SCORE:-NA}

if [ "$SCORE" = "NA" ]; then
  RUN_DIR="data/output/run4_no_score_job${SLURM_JOB_ID}"
else
  SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')
  RUN_DIR="data/output/run4_${SCORE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$RUN_DIR"

cp "$PRED_CSV" "$RUN_DIR/" || true
cp "src/run4_prompts.py" "$RUN_DIR/prompts_used.py" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

if [ -d "$TEMP_RUN_DIR/eval" ]; then
  mv "$TEMP_RUN_DIR/eval" "$RUN_DIR/eval"
fi

rmdir "$TEMP_RUN_DIR" 2>/dev/null || true

echo "Archived in: $RUN_DIR"
echo "Score: ${SCORE}%"
