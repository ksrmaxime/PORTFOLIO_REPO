#!/bin/bash -l
#SBATCH --job-name=merge_target_outputs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/merge_target_outputs_%j.out
#SBATCH --error=logs/merge_target_outputs_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed

# Fusionne plusieurs fichiers de sortie run_target, chacun ne portant qu'un
# sous-ensemble des 10 colonnes target_<CODE> sur le même corpus sous-jacent
# (ex. deux chaînes sbatch_run_target.sh lancées séparément, l'une couvrant
# les cibles 1-5, l'autre 6-10), en un seul fichier portant les 10 colonnes.
# C'est l'entrée requise par sbatch_run_target_control.sh.
#
# $1, $2, ... = chemins des fichiers run_target à fusionner (>= 2, parquet ou csv)
if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch sbatch_merge_target_outputs.sh <input1.parquet> <input2.parquet> [...]" >&2
  exit 1
fi

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "DATE=$(date -Is)"
echo "Inputs: $*"

OUTPUT="${OUTDIR}/laws_structure_with_target_merged_job${SLURM_JOB_ID}.parquet"

python scripts/merge_target_outputs.py \
  --inputs "$@" \
  --output "$OUTPUT"

RUN_DIR="data/output/merge_target_outputs_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cp "$OUTPUT" "$RUN_DIR/" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

echo "Archived in: $RUN_DIR"
echo "Merged file: $OUTPUT"
echo
echo "Pour lancer le run de contrôle sur ce fichier :"
echo "  sbatch sbatch_run_target_control.sh 1 $OUTPUT"
