#!/bin/bash
#SBATCH --job-name=pf_run2_relart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/run2_relevant_art_%j.out
#SBATCH --error=logs/run2_relevant_art_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load python/3.12.1

echo "=== SLURM ==="
echo "JOBID=$SLURM_JOB_ID HOST=$(hostname) DATE=$(date -Iseconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

cd /work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
source .venv/bin/activate

OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed
mkdir -p logs "$OUTDIR"

python -V

# Optional: quick GPU info
nvidia-smi || true

python scripts/run2_relevant_art_llm.py