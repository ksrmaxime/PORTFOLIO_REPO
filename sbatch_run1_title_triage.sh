#!/bin/bash
#SBATCH --job-name=run1_title_triage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run1_%j.out
#SBATCH --error=logs/run1_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL


# ---- safety ----
set -euo pipefail

echo "Job started on $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# ---- modules ----
module purge
module load python/3.12.1

# ---- paths ----
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
SCRATCHDIR=/scratch/mkaiser3

cd $WORKDIR

# ---- venv ----
source .venv/bin/activate

# ---- sanity checks ----
python - << 'EOF'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
EOF

# ---- run ----
python scripts/llm_title_triage_run1.py \
  --platform curnagl \
  --workdir $WORKDIR \
  --scratchdir $SCRATCHDIR \
  --backend transformers \
  --model-path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16

echo "Job finished"
