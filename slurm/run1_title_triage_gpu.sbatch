#!/bin/bash
#SBATCH -J fedlex_run1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/mkaiser3/logs/%x_%j.out
#SBATCH -e /scratch/mkaiser3/logs/%x_%j.err

set -euo pipefail

WORK=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
SCR=/scratch/mkaiser3

cd "$WORK"

source .venv/bin/activate

python scripts/llm_title_triage_run1.py \
  --platform curnagl \
  --workdir "$WORK" \
  --scratchdir "$SCR" \
  --backend vllm \
  --model-path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --chunk-size 80 \
  --max-tokens 800 \
  --temperature 0.0
