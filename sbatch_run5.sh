#!/bin/bash -l
#SBATCH --job-name=run5_portfolio_coding
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run5_portfolio_coding_%j.out
#SBATCH --error=logs/run5_portfolio_coding_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed
OUTBASE="/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure_with_portfolio"

# Input = output du run 4 (remplacer JOB_ID par le job ID du run 4)
RUN4_JOB_ID=63674616
INPUT="${OUTDIR}/laws_structure_with_ai_confirmed_job${RUN4_JOB_ID}.parquet"

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

python scripts/run5_pipeline.py \
  --input  "$INPUT" \
  --output_base "$OUTBASE" \
  --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --trust_remote_code \
  --text_col text \
  --final_ai_relevant_col final_ai_relevant \
  --batch_size 8 \
  --max_new_tokens 400 \
  --temperature 0.0

# --- Évaluation ---
# NB: nécessite que PORTFOLIO_GOLD.csv contienne des colonnes "Targets" et
# "Instruments" avec des codes séparés par "|" (mêmes codes que src/run5_prompts.py).
# Tant que ce codage manuel n'existe pas, ce bloc échouera silencieusement sur le
# score (colonnes manquantes) — ajuster --rename_gold_cols / --list_seps une fois
# la gold data construite (cf. section 3.3 étape 1 du PoC).
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"
ENTRIES_CSV="${OUTBASE}_entries_job${SLURM_JOB_ID}.csv"

GOLD_CSV="data/external/PORTFOLIO_GOLD.csv"

TEMP_RUN_DIR="data/output/run5_job${SLURM_JOB_ID}"
mkdir -p "$TEMP_RUN_DIR"

if [ -f "$GOLD_CSV" ] && python -c "
import pandas as pd
g = pd.read_csv('$GOLD_CSV')
import sys
sys.exit(0 if {'Targets','Instruments'}.issubset(g.columns) else 1)
"; then
  GOLD_WITH_ID="${TEMP_RUN_DIR}/gold_with_row_id.csv"
  python scripts/add_row_id.py "$GOLD_CSV" --col row_id --overwrite --out "$GOLD_WITH_ID"

  SCORE_LOG=$(python scripts/score.py \
    --pred "$PRED_CSV" \
    --gold "$GOLD_WITH_ID" \
    --id_col row_id \
    --cols TARGET_CODES,INSTRUMENT_CODES \
    --col_kinds TARGET_CODES=list,INSTRUMENT_CODES=list \
    --list_seps TARGET_CODES=\|,INSTRUMENT_CODES=\| \
    --rename_gold_cols Targets=TARGET_CODES,Instruments=INSTRUMENT_CODES \
    --extra_cols text \
    --report_dir "$TEMP_RUN_DIR/eval")

  echo "$SCORE_LOG"

  SCORE=$(echo "$SCORE_LOG" | awk '/^Similarity:/ {gsub(/%/,"",$2); print $2; exit}')
else
  echo "Gold data absent ou incomplète (colonnes Targets/Instruments) — évaluation sautée."
  SCORE=""
fi

SCORE=${SCORE:-NA}

if [ "$SCORE" = "NA" ]; then
  RUN_DIR="data/output/run5_no_score_job${SLURM_JOB_ID}"
else
  SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')
  RUN_DIR="data/output/run5_${SCORE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$RUN_DIR"

cp "$PRED_CSV" "$RUN_DIR/" || true
cp "$ENTRIES_CSV" "$RUN_DIR/" || true
cp "src/run5_prompts.py" "$RUN_DIR/prompts_used.py" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

if [ -d "$TEMP_RUN_DIR/eval" ]; then
  mv "$TEMP_RUN_DIR/eval" "$RUN_DIR/eval"
fi

rm -rf "$TEMP_RUN_DIR" 2>/dev/null || true

echo "Archived in: $RUN_DIR"
echo "Score: ${SCORE}%"
