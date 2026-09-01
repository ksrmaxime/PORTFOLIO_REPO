#!/bin/bash -l
#SBATCH --job-name=run_target
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_target_%j.out
#SBATCH --error=logs/run_target_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed

# Cette chaîne tourne en PREMIER, sur tout le corpus AI-relevant (même input par
# défaut que sbatch_run_inst.sh). Une fois les 10 étapes terminées, lancer
# manuellement sbatch_run_inst.sh en lui passant en argument le job id SLURM de
# cette dernière étape (HIGH_STAKES_RISKS) : il filtrera automatiquement
# aux articles ayant au moins une cible = True avant de coder les instruments.
#
# --- Chaîne de cibles : run_target_1 = RESEARCH_INNOVATION, run_target_2 = SKILLS_HUMAN_CAPITAL, etc. ---
# Doit rester synchronisé (mêmes codes, même ordre) avec TARGET_DEFINITIONS
# dans src/run_target_prompts.py (nouvelle taxonomie à 10 cibles : les
# anciennes HIGH_STAKES_RIGHTS + INFORMATION_SOCIETAL_HARMS sont fusionnées
# en HIGH_STAKES_RISKS, et EXPERIMENTATION_MARKET est fusionnée dans
# ADOPTION_DIFFUSION).
TARGETS=(RESEARCH_INNOVATION SKILLS_HUMAN_CAPITAL DATA_ACCESS_RESOURCES COMPUTE_INFRASTRUCTURE ADOPTION_DIFFUSION DATA_PRIVACY IP_CREATIVE_RIGHTS SECURITY_ROBUSTNESS ACCOUNTABILITY_TRANSPARENCY HIGH_STAKES_RISKS)
N_TARGETS=${#TARGETS[@]}

# $1 = index de cible (1-12), $2 = fichier d'entrée (optionnel, seulement utile pour l'index 1)
IDX="${1:?Usage: sbatch sbatch_run_target.sh <target_index 1-12> [input_file]}"
if [ "$IDX" -lt 1 ] || [ "$IDX" -gt "$N_TARGETS" ]; then
  echo "IDX must be between 1 and $N_TARGETS, got: $IDX" >&2
  exit 1
fi
CODE="${TARGETS[$((IDX - 1))]}"
CODE_LOWER=$(echo "$CODE" | tr '[:upper:]' '[:lower:]')

DEFAULT_INPUT="/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed/laws_structure_selected_with_ai_relevant.parquet"
INPUT="${2:-$DEFAULT_INPUT}"

OUTBASE="${OUTDIR}/laws_structure_with_target_${CODE_LOWER}"

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
echo "Target: $IDX/$N_TARGETS = $CODE"
echo "Input: $INPUT"
nvidia-smi -L || true

python scripts/run_target_pipeline.py \
  --input  "$INPUT" \
  --output_base "$OUTBASE" \
  --target_code "$CODE" \
  --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --trust_remote_code \
  --text_col text \
  --level_col level \
  --batch_size 8 \
  --max_new_tokens 150 \
  --temperature 0.0

# --- Évaluation ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"
PRED_PARQUET="${OUTBASE}_job${SLURM_JOB_ID}.parquet"

DECISION_COL="target_${CODE}"

# fichier benchmark gold (colonnes Target_<CODE> attendues, une par cible)
GOLD_CSV="data/external/PORTFOLIO_GOLD.csv"
GOLD_COL="Target_${CODE}"

TEMP_RUN_DIR="data/output/run_target_${IDX}_${CODE}_job${SLURM_JOB_ID}"
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
  RUN_DIR="data/output/run_target_${IDX}_${CODE}_no_score_job${SLURM_JOB_ID}"
else
  SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')
  RUN_DIR="data/output/run_target_${IDX}_${CODE}_${SCORE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$RUN_DIR"

# --- Archive: outputs + prompt ---
cp "$PRED_CSV" "$RUN_DIR/" || true
cp "src/run_target_prompts.py" "$RUN_DIR/prompts_used.py" || true
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
  echo "Chaining: submitting target $NEXT_IDX/$N_TARGETS = $NEXT_CODE"
  sbatch --job-name="run_target_${NEXT_IDX}_${NEXT_CODE}" sbatch_run_target.sh "$NEXT_IDX" "$PRED_PARQUET"
else
  echo "Dernière cible ($CODE) traitée — fin de la chaîne run_target."
fi
