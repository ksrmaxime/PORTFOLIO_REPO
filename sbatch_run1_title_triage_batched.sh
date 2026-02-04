#!/bin/bash
# sbatch/sbatch_run1_title_triage_batched.sh
#SBATCH --job-name=run1_title_batched
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run1_batched_%j.out
#SBATCH --error=logs/run1_batched_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
SCRATCHDIR=/scratch/mkaiser3

cd "$WORKDIR"
source .venv/bin/activate

echo "=== SLURM ==="
echo "JOBID=$SLURM_JOB_ID HOST=$(hostname) PARTITION=$SLURM_JOB_PARTITION"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== NVIDIA-SMI ==="
which nvidia-smi || true
nvidia-smi || true

echo "=== PYTORCH CUDA ==="
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("cuda visible devices env:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
PY


mkdir -p logs "$SCRATCHDIR/portfolio/run1"

# GPU utilization logging (post-mortem)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 30 > "logs/gpu_${SLURM_JOB_ID}.csv" &
NSMI_PID=$!

python scripts/run1_title_triage_batched.py \
  --workdir "$WORKDIR" \
  --scratchdir "$SCRATCHDIR" \
  --model-path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --items-per-prompt 40 \
  --prompts-per-batch 8 \
  --max-tokens 160 \
  --temperature 0.0

kill "$NSMI_PID" || true
echo "Done."
