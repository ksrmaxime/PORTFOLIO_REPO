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

python scripts/run_pipeline_run2.py \
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

# capture du score (ligne: "Similarity: 51.08%")
SCORE=$(python scripts/evaluate_vs_gold.py \
  --pred "$PRED_CSV" \
  --gold "$GOLD_CSV" \
  --use_row_order \
  --id_col __row__ \
  --cols RELEVANT_ART \
  | awk '/Similarity:/ {gsub(/%/,"",$2); print $2}')

# normaliser pour nom de dossier (51.08 -> 51p08)
SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')

RUN_DIR="data/output/run2_${SCORE_TAG}"
mkdir -p "$RUN_DIR"

# --- Archive: outputs + prompt ---
cp "$PRED_CSV" "$RUN_DIR/"
cp "src/prompts.py" "$RUN_DIR/prompts_used.py"
cp "$0" "$RUN_DIR/sbatch_used.sbatch"

echo "Archived in: $RUN_DIR"
echo "Score: ${SCORE}%"