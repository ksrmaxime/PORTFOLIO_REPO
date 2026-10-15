#!/bin/bash -l
#SBATCH --job-name=fedlex_kw_filter
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fedlex_kw_filter_%j.out
#SBATCH --error=logs/fedlex_kw_filter_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Filtre par mots-clés (IA & synonymes, cf. src/keyword_filter.py) du dataset produit par
# sbatch_download.sh : garde les actes dont au moins un article / titre de chapitre
# contient un mot-clé, avec toutes leurs lignes.
#
# Usage :
#   sbatch sbatch_keyword_filter.sh                   # entrée par défaut : fedlex/all
#   sbatch sbatch_keyword_filter.sh interne           # sortie d'un download "interne"
#   TIERS=core sbatch sbatch_keyword_filter.sh        # mots-clés IA explicites seulement
#   INPUT=/chemin/fichier.parquet sbatch sbatch_keyword_filter.sh

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

DOC_TYPES="${1:-all}"
TIERS="${TIERS:-core,extended}"

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
INDIR="${WORKDIR}/data/processed/fedlex/${DOC_TYPES}"
INPUT="${INPUT:-${INDIR}/laws_structure_final.parquet}"
TIER_TAG=$(echo "$TIERS" | tr ',' '_')
OUTBASE="${INDIR}/keyword_filter/laws_kw_${TIER_TAG}"

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"
echo "INPUT=$INPUT"
echo "OUTBASE=$OUTBASE TIERS=$TIERS"
echo "GIT=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

if [ ! -f "$INPUT" ]; then
  echo "Fichier d'entrée introuvable : $INPUT (lancer d'abord sbatch_download.sh)" >&2
  exit 1
fi

python scripts/keyword_filter.py \
  --input "$INPUT" \
  --output_base "$OUTBASE" \
  --tiers "$TIERS"
