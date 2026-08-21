#!/bin/bash -l
#SBATCH --job-name=run6_portfolio_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/run6_portfolio_plot_%j.out
#SBATCH --error=logs/run6_portfolio_plot_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/output/portfolio
OUTBASE="${OUTDIR}/portfolio_final"

# =============================================================================
# Fichier d'entrée = sortie "entries" du run5 (table longue article x cible x
# instrument, cf. scripts/run5_pipeline.py). Deux façons de le sélectionner :
#
#   Option A (par défaut) : indiquer le job ID SLURM du run5 dont on veut
#   visualiser la sortie. Le chemin est reconstruit automatiquement à partir
#   du même OUTBASE que sbatch_run5.sh.
RUN5_JOB_ID=00000000

#   Option B : donner directement le chemin complet du fichier
#   "*_entries_job*.parquet" (ou .csv) à utiliser. Si non vide, cette valeur
#   est prioritaire sur RUN5_JOB_ID ci-dessus.
INPUT_OVERRIDE=""
# =============================================================================

RUN5_OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed
RUN5_OUTBASE="${RUN5_OUTDIR}/laws_structure_with_portfolio"
DEFAULT_INPUT="${RUN5_OUTBASE}_entries_job${RUN5_JOB_ID}.parquet"

if [ -n "$INPUT_OVERRIDE" ]; then
  INPUT="$INPUT_OVERRIDE"
else
  INPUT="$DEFAULT_INPUT"
fi

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "DATE=$(date -Is)"
echo "INPUT=$INPUT"

if [ ! -f "$INPUT" ]; then
  echo "ERREUR: fichier d'entrée introuvable: $INPUT" >&2
  echo "-> régler RUN5_JOB_ID (ou INPUT_OVERRIDE) en tête de ce script." >&2
  exit 1
fi

python scripts/run6_pipeline.py \
  --input "$INPUT" \
  --output_base "${OUTBASE}_job${SLURM_JOB_ID}" \
  --title "AI Regulation Portfolio — Switzerland (run5 job ${RUN5_JOB_ID})"

RUN_DIR="output/portfolio/run6_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cp "${OUTBASE}_job${SLURM_JOB_ID}".* "$RUN_DIR/" 2>/dev/null || true
cp "src/run6_plot.py" "$RUN_DIR/run6_plot_used.py" || true
cp "src/run6_config.py" "$RUN_DIR/run6_config_used.py" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

echo "Archived in: $RUN_DIR"
