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

# CRITICAL: force import from repo/src (so you don't use an old installed package)
export PYTHONPATH="/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/src:${PYTHONPATH:-}"

python -V
python - <<'PY'
import inspect
import portfolio_repo.llm.curnagl_client as m
print("curnagl_client imported from:", m.__file__)
print("has TransformersClient:", hasattr(m, "TransformersClient"))
PY

nvidia-smi || true

python scripts/run2_relevant_art_llm.py