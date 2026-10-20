#!/bin/bash -l
#SBATCH --job-name=llm1_ai_relevance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/llm1_ai_relevance_%j.out
#SBATCH --error=logs/llm1_ai_relevance_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Step LLM 1 (v3) : pour chaque article (level 6) contenant un mot-clé, classement
# SPECIFIQUE / NEUTRE / NON (cf. src/ai_relevance_prompts.py) :
#   - IA nommée explicitement -> SPECIFIQUE par règle, sans LLM ;
#   - sinon Apertus répond A/B/C (probabilité des trois lettres, une passe par article) ;
#     le prompt inclut la définition du terme si l'acte en donne une ; justification d'une phrase ;
#   - propagation intra-acte des SPECIFIQUE pour les termes qui désignent un système.
#
# Usage :
#   sbatch sbatch_ai_relevance.sh                       # entrée par défaut (filtre core+extended sur fedlex/all)
#   LIMIT=10 sbatch sbatch_ai_relevance.sh              # test rapide sur 10 articles
#   GOLD=/chemin/PORTFOLIO_NEW_GOLD.csv sbatch sbatch_ai_relevance.sh   # + comparaison au codage manuel
#   INPUT=/chemin/…_annotated.parquet sbatch sbatch_ai_relevance.sh
#   CALIBRATE=1 sbatch sbatch_ai_relevance.sh           # label sur probabilités calibrées (défaut : brutes)
#   NO_PROPAGATION=1 / NO_DEFINITIONS=1 sbatch …        # désactiver la propagation / les définitions

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
KWDIR="${WORKDIR}/data/processed/fedlex/all/keyword_filter"
INPUT="${INPUT:-${KWDIR}/laws_structure_kw_core_extended_annotated.parquet}"
OUTBASE="${WORKDIR}/data/processed/fedlex/all/llm1_ai_relevance/ai_relevance"
MODEL_PATH="${MODEL_PATH:-/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509}"
LIMIT="${LIMIT:-}"
GOLD="${GOLD:-}"
CALIBRATE="${CALIBRATE:-}"
NO_PROPAGATION="${NO_PROPAGATION:-}"
NO_DEFINITIONS="${NO_DEFINITIONS:-}"

cd "$WORKDIR"
source .venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"
echo "INPUT=$INPUT"
echo "MODEL=$MODEL_PATH LIMIT=${LIMIT:-tous} GOLD=${GOLD:-aucun}"
echo "GIT=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi -L || true

if [ ! -f "$INPUT" ]; then
  echo "Fichier d'entrée introuvable : $INPUT (lancer d'abord sbatch_keyword_filter.sh)" >&2
  exit 1
fi

EXTRA_ARGS=()
[ -n "$LIMIT" ] && EXTRA_ARGS+=(--limit "$LIMIT")
[ -n "$GOLD" ] && EXTRA_ARGS+=(--gold "$GOLD")
[ -n "$CALIBRATE" ] && EXTRA_ARGS+=(--calibrate)
[ -n "$NO_PROPAGATION" ] && EXTRA_ARGS+=(--no_propagation)
[ -n "$NO_DEFINITIONS" ] && EXTRA_ARGS+=(--no_definitions)

python scripts/run_ai_relevance.py \
  --input "$INPUT" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype bf16 \
  --trust_remote_code \
  --batch_size 4 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# archive du prompt utilisé à côté des résultats
cp src/ai_relevance_prompts.py "$(dirname "$OUTBASE")/prompts_used_job${SLURM_JOB_ID:-nojobid}.py" || true
