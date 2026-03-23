#!/bin/bash
#SBATCH --job-name=run2_relart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run2_relart_%j.out
#SBATCH --error=logs/run2_relart_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
SCRATCHDIR=/scratch/mkaiser3
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed
OUTBASE="/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure_with_relart"

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

python scripts/run2_pipeline.py \
  --input  "/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure_selected_with_ai_relevant.parquet" \
  --output_base "$OUTBASE" \
  --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --trust_remote_code \
  --text_col text \
  --level_col level \
  --decision_col RELEVANT_ART \
  --justif_col  RELEVANT_ART_JUSTIF \
  --batch_size 32 \
  --max_new_tokens 160 \
  --temperature 0.0

# --- Évaluation ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"

# ton fichier "gold" (humain) ici:
GOLD_CSV="data/external/RUN2_GOLD.csv"

# dossier temporaire pour l'évaluation
TEMP_RUN_DIR="data/output/run2_job${SLURM_JOB_ID}"
mkdir -p "$TEMP_RUN_DIR"

# Ajouter row_id au gold (copie temporaire pour ne pas modifier l'original)
GOLD_WITH_ID="${TEMP_RUN_DIR}/gold_with_row_id.csv"
python scripts/add_row_id.py "$GOLD_CSV" --col row_id --out "$GOLD_WITH_ID"

# run evaluation and capture stdout
SCORE_LOG=$(python scripts/score.py \
  --pred "$PRED_CSV" \
  --gold "$GOLD_WITH_ID" \
  --id_col row_id \
  --cols RELEVANT_ART \
  --col_kinds RELEVANT_ART=label \
  --report_dir "$TEMP_RUN_DIR/eval")

echo "$SCORE_LOG"

# extract numeric similarity from stdout
SCORE=$(echo "$SCORE_LOG" | awk '/^Similarity:/ {gsub(/%/,"",$2); print $2; exit}')
SCORE=${SCORE:-NA}

# normaliser pour nom de dossier (51.08 -> 51p08)
if [ "$SCORE" = "NA" ]; then
  RUN_DIR="data/output/run2_no_score_job${SLURM_JOB_ID}"
else
  SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')
  RUN_DIR="data/output/run2_${SCORE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$RUN_DIR"

# --- Archive: outputs + prompt ---
cp "$PRED_CSV" "$RUN_DIR/" || true
cp "src/run2_prompts.py" "$RUN_DIR/prompts_used.py" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

# move eval reports
if [ -d "$TEMP_RUN_DIR/eval" ]; then
  mv "$TEMP_RUN_DIR/eval" "$RUN_DIR/eval"
fi

# cleanup temp dir if empty
rmdir "$TEMP_RUN_DIR" 2>/dev/null || true

echo "Archived in: $RUN_DIR"
echo "Score: ${SCORE}%"