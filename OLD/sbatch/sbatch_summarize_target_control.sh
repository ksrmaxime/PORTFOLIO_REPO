#!/bin/bash -l
#SBATCH --job-name=summarize_target_control
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/summarize_target_control_%j.out
#SBATCH --error=logs/summarize_target_control_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed

# Prend la sortie de la DERNIÈRE étape de la chaîne sbatch_run_target_control.sh
# (cible SOCIETAL_HARMS, IDX=10 — son fichier porte les 10 colonnes
# control_target_<CODE>) et produit un récapitulatif des articles toujours
# classés True après contrôle, avec la ou les cibles concernées.
#
# $1 = job id SLURM de la dernière étape run_target_control terminée (celle
#      pour SOCIETAL_HARMS), ou alternativement un chemin de fichier
#      parquet/csv déjà produit par cette dernière étape.
SECOND_ARG="${1:?Usage: sbatch sbatch_summarize_target_control.sh <last_run_target_control_job_id|input_file>}"

if [[ "$SECOND_ARG" =~ ^[0-9]+$ ]]; then
  CONTROL_JOB_ID="$SECOND_ARG"
  INPUT="${OUTDIR}/laws_structure_with_target_control_societal_harms_job${CONTROL_JOB_ID}.parquet"
  if [ ! -f "$INPUT" ]; then
    echo "Fichier de sortie run_target_control introuvable: $INPUT (job id incorrect, ou ce n'était pas la dernière étape de la chaîne ?)" >&2
    exit 1
  fi
else
  INPUT="$SECOND_ARG"
fi

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "DATE=$(date -Is)"
echo "Input: $INPUT"

OUTBASE="${OUTDIR}/target_control_summary_job${SLURM_JOB_ID}"

python scripts/summarize_target_control.py \
  --input "$INPUT" \
  --output_base "$OUTBASE"

RUN_DIR="data/output/summarize_target_control_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cp "${OUTBASE}_wide.csv" "${OUTBASE}_long.csv" "$RUN_DIR/" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

echo "Archived in: $RUN_DIR"
