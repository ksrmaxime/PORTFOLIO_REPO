#!/bin/bash -l
#SBATCH --job-name=llm2_policy_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/llm2_policy_class_%j.out
#SBATCH --error=logs/llm2_policy_class_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Step LLM 2 : cible (cadran + sous-catégorie) et instrument des articles classés SPECIFIQUE
# à l'étape 1 (cf. src/policy_classification_prompts.py et le codebook).
# Trois questions par article, réponse en une lettre lue par probabilité :
#   Q1 cadran (ou « aucun mécanisme ») -> Q2 sous-catégorie -> Q3 instrument.
#   Cadrans « risque » : question oui/non sur l'accountability avant la sous-catégorie.
#   DEBIAS=1 : poser aussi cadran / sous-catégorie / instrument dans deux ordres.
#
# Usage :
#   sbatch sbatch_policy_classification.sh                          # dernière sortie de l'étape 1
#   INPUT=/chemin/ai_relevance_job66295864.parquet sbatch sbatch_policy_classification.sh
#   GOLD=/chemin/PORTFOLIO_NEW_GOLD.csv sbatch sbatch_policy_classification.sh   # + comparaison au gold
#   LIMIT=5 sbatch sbatch_policy_classification.sh                  # test rapide
#   GOLD=… FROM_GOLD=1 sbatch sbatch_policy_classification.sh        # classer les SPECIFIQUE du gold (évalue l'étape 2 seule)

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
LLM1DIR="${WORKDIR}/data/processed/fedlex/all/llm1_ai_relevance"
OUTBASE="${WORKDIR}/data/processed/fedlex/all/llm2_policy_classification/policy_class"
MODEL_PATH="${MODEL_PATH:-/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509}"
LIMIT="${LIMIT:-}"
GOLD="${GOLD:-}"
NO_DEFINITIONS="${NO_DEFINITIONS:-}"
FROM_GOLD="${FROM_GOLD:-}"
DEBIAS="${DEBIAS:-}"

cd "$WORKDIR"
source .venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs

# entrée par défaut : la sortie la plus récente de l'étape 1
if [ -z "${INPUT:-}" ]; then
  INPUT=$(ls -t "${LLM1DIR}"/ai_relevance_job*.parquet 2>/dev/null | grep -v _articles | head -n 1 || true)
fi
if [ -z "${INPUT:-}" ] || [ ! -f "$INPUT" ]; then
  echo "Entrée introuvable (${INPUT:-aucune}) : lancer d'abord sbatch_ai_relevance.sh ou préciser INPUT=" >&2
  exit 1
fi

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"
echo "INPUT=$INPUT"
echo "MODEL=$MODEL_PATH LIMIT=${LIMIT:-tous} GOLD=${GOLD:-aucun}"
echo "GIT=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi -L || true

EXTRA_ARGS=()
[ -n "$LIMIT" ] && EXTRA_ARGS+=(--limit "$LIMIT")
[ -n "$GOLD" ] && EXTRA_ARGS+=(--gold "$GOLD")
[ -n "$NO_DEFINITIONS" ] && EXTRA_ARGS+=(--no_definitions)
[ -n "$FROM_GOLD" ] && EXTRA_ARGS+=(--select_from_gold)
[ -n "$DEBIAS" ] && EXTRA_ARGS+=(--debias)

python scripts/run_policy_classification.py \
  --input "$INPUT" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype bf16 \
  --trust_remote_code \
  --batch_size 4 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

cp src/policy_classification_prompts.py "$(dirname "$OUTBASE")/prompts_used_job${SLURM_JOB_ID:-nojobid}.py" || true
