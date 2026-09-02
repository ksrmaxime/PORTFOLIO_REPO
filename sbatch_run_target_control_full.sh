#!/bin/bash -l
#SBATCH --job-name=run_target_control_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_target_control_full_%j.out
#SBATCH --error=logs/run_target_control_full_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed

# Variante EXPLORATOIRE de sbatch_run_target_control.sh : au lieu de reprendre
# uniquement les articles classés OUI par run_target (colonne target_<CODE>),
# cette chaîne tourne en PREMIER, directement sur tout le corpus AI-relevant
# (même input par défaut que sbatch_run_target.sh) et envoie TOUS les
# articles au prompt de contrôle, cible par cible.
#
# Le prompt lui-même (src/run_target_control_prompts.py) N'EST PAS modifié :
# il continue d'affirmer à chaque article "la mesure ci-dessus a été
# classifiée par un autre modèle comme répondant à ce problème public",
# que ce soit vrai ou non. L'idée est de voir si ce cadrage (second avis,
# définition courte) donne de meilleures classifications que le prompt
# détaillé de run_target quand on l'applique tel quel à tout le corpus,
# sans présupposer le filtre.
#
# --- Doit rester synchronisé (mêmes codes, même ordre) avec TARGETS dans
# sbatch_run_target.sh et TARGET_DEFINITIONS dans src/run_target_prompts.py.
TARGETS=(RESEARCH_INNOVATION SKILLS_HUMAN_CAPITAL DATA_ACCESS_RESOURCES COMPUTE_INFRASTRUCTURE ADOPTION_DIFFUSION DATA_PRIVACY IP_CREATIVE_RIGHTS SECURITY_ROBUSTNESS ACCOUNTABILITY_TRANSPARENCY HIGH_STAKES_RISKS)
N_TARGETS=${#TARGETS[@]}

# $1 = index de cible (1-10), $2 = fichier d'entrée (optionnel, seulement utile pour l'index 1)
IDX="${1:?Usage: sbatch sbatch_run_target_control_full.sh <target_index 1-10> [input_file]}"
if [ "$IDX" -lt 1 ] || [ "$IDX" -gt "$N_TARGETS" ]; then
  echo "IDX must be between 1 and $N_TARGETS, got: $IDX" >&2
  exit 1
fi
CODE="${TARGETS[$((IDX - 1))]}"
CODE_LOWER=$(echo "$CODE" | tr '[:upper:]' '[:lower:]')

DEFAULT_INPUT="/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure_selected_with_ai_relevant.parquet"
INPUT="${2:-$DEFAULT_INPUT}"

OUTBASE="${OUTDIR}/laws_structure_with_target_control_full_${CODE_LOWER}"

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
echo "Control-full target: $IDX/$N_TARGETS = $CODE"
echo "Input: $INPUT"
nvidia-smi -L || true

python scripts/run_target_control_pipeline.py \
  --input  "$INPUT" \
  --output_base "$OUTBASE" \
  --target_code "$CODE" \
  --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --trust_remote_code \
  --text_col text \
  --level_col level \
  --send_all \
  --batch_size 8 \
  --max_new_tokens 150 \
  --temperature 0.0

# --- Évaluation contre le gold (comme sbatch_run_target.sh) ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"
PRED_PARQUET="${OUTBASE}_job${SLURM_JOB_ID}.parquet"

DECISION_COL="control_target_${CODE}"

GOLD_CSV="data/external/PORTFOLIO_GOLD.csv"
GOLD_COL="Target_${CODE}"

TEMP_RUN_DIR="data/output/run_target_control_full_${IDX}_${CODE}_job${SLURM_JOB_ID}"
mkdir -p "$TEMP_RUN_DIR"

if [ -f "$GOLD_CSV" ] && python -c "
import pandas as pd
g = pd.read_csv('$GOLD_CSV')
import sys
sys.exit(0 if '$GOLD_COL' in g.columns else 1)
"; then
  GOLD_WITH_ID="${TEMP_RUN_DIR}/gold_with_row_id.csv"
  python scripts/add_row_id.py "$GOLD_CSV" --col row_id --overwrite --out "$GOLD_WITH_ID"

  SCORE_LOG=$(python scripts/score.py \
    --pred "$PRED_CSV" \
    --gold "$GOLD_WITH_ID" \
    --id_col row_id \
    --cols "$DECISION_COL" \
    --col_kinds "${DECISION_COL}=label" \
    --rename_gold_cols "${GOLD_COL}=${DECISION_COL}" \
    --extra_cols text \
    --report_dir "$TEMP_RUN_DIR/eval")

  echo "$SCORE_LOG"

  SCORE=$(echo "$SCORE_LOG" | awk '/^Similarity:/ {gsub(/%/,"",$2); print $2; exit}')
else
  echo "Gold data absent ou colonne '$GOLD_COL' manquante — évaluation sautée."
  SCORE=""
fi

SCORE=${SCORE:-NA}

if [ "$SCORE" = "NA" ]; then
  RUN_DIR="data/output/run_target_control_full_${IDX}_${CODE}_no_score_job${SLURM_JOB_ID}"
else
  SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')
  RUN_DIR="data/output/run_target_control_full_${IDX}_${CODE}_${SCORE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$RUN_DIR"

# --- Archive: outputs + prompt ---
cp "$PRED_CSV" "$RUN_DIR/" || true
cp "src/run_target_control_prompts.py" "$RUN_DIR/prompts_used.py" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

if [ -d "$TEMP_RUN_DIR/eval" ]; then
  mv "$TEMP_RUN_DIR/eval" "$RUN_DIR/eval"
fi

rmdir "$TEMP_RUN_DIR" 2>/dev/null || true

echo "Archived in: $RUN_DIR"
echo "Score: ${SCORE}%"

# --- Chaînage : soumet automatiquement la cible suivante ---
# N'est atteint que si le run ci-dessus a réussi (set -e). Le fichier produit par ce
# stage devient l'entrée du stage suivant.
if [ "$IDX" -lt "$N_TARGETS" ]; then
  NEXT_IDX=$((IDX + 1))
  NEXT_CODE="${TARGETS[$((NEXT_IDX - 1))]}"
  echo "Chaining: submitting control-full target $NEXT_IDX/$N_TARGETS = $NEXT_CODE"
  sbatch --job-name="run_target_control_full_${NEXT_IDX}_${NEXT_CODE}" sbatch_run_target_control_full.sh "$NEXT_IDX" "$PRED_PARQUET"
else
  echo "Dernière cible ($CODE) traitée — fin de la chaîne run_target_control_full."
fi
