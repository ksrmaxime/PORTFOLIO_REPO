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

mkdir -p logs "$SCRATCHDIR/portfolio/run1"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"

echo "=== GPU (start) ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader || true

echo "=== PYTORCH CUDA ==="
python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total_vram_GB:", round(props.total_memory / (1024**3), 2))
PY

# --- GPU monitoring (VRAM + util), every 30s, written to logs/ ---
GPU_CSV="logs/gpu_${SLURM_JOB_ID}.csv"

cleanup() {
  if [[ -n "${NSMI_PID:-}" ]]; then
    kill "${NSMI_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "=== GPU monitor ==="
echo "Writing: ${GPU_CSV}"
(
  echo "timestamp,util.gpu,util.mem,mem.used,mem.total,power.draw"
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
      --format=csv,noheader
    sleep 30
  done
) > "${GPU_CSV}" &
NSMI_PID=$!

echo "=== RUN ==="
echo "WORKDIR=${WORKDIR}"
echo "SCRATCHDIR=${SCRATCHDIR}"

python scripts/run1_title_triage_batched.py \
  --workdir "$WORKDIR" \
  --scratchdir "$SCRATCHDIR" \
  --model-path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --items-per-prompt 40 \
  --prompts-per-batch 8 \
  --max-tokens 160 \
  --temperature 0.0

echo "=== GPU (end) ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader || true

echo "Done."

