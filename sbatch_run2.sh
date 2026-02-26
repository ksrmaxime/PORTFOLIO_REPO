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

NAS_DIR=/nas/FAC/FDCA/IDHEAP/mhinterl/parp/D2c/maxime/output/PORTFOLIO_RUN2
mkdir -p "$NAS_DIR"

cp "${OUTBASE}_job${SLURM_JOB_ID}.csv" "$NAS_DIR"

echo "Copied results to NAS"