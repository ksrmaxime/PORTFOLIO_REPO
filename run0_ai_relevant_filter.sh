#!/bin/bash
#SBATCH --job-name=pf_ai_relevant
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/run0_ai_relevant_%j.out
#SBATCH --error=logs/run0_ai_relevant_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load python/3.12.1

echo "=== SLURM ==="
echo "JOBID=$SLURM_JOB_ID HOST=$(hostname) DATE=$(date -Iseconds)"

# Adapter si ton environnement s'appelle autrement
# (si tu as déjà l'habitude d'activer .venv dans le repo, garde ça)
cd /work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
source .venv/bin/activate

OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed
mkdir -p logs "$OUTDIR"

python -V
python scripts/run0_ai_relevant_filter.py