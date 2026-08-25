#!/bin/bash -l
#SBATCH --job-name=run_inst
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_inst_%j.out
#SBATCH --error=logs/run_inst_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed

# --- Chaîne d'instruments : run_inst_1 = VOLUNTARY, run_inst_2 = TAXES_SUBSIDIES, etc. ---
INSTRUMENTS=(VOLUNTARY TAXES_SUBSIDIES PUBLIC_INVESTMENT PROHIBITION_BAN PLANNING_EVALUATION OBLIGATION LIABILITY)
N_INSTRUMENTS=${#INSTRUMENTS[@]}

# Cette chaîne se lance manuellement APRÈS la chaîne run_target (sbatch_run_target.sh,
# 12 étapes) : run_target tourne d'abord sur tout le corpus AI-relevant, puis run_inst
# ne tourne que sur les articles retenus par au moins une cible (~1% du corpus), au
# lieu de tout le corpus. C'est plus efficace car quasi tous les articles ont un
# instrument, alors que très peu sont pertinents pour une cible donnée.
#
# $1 = index d'instrument (1-7)
# $2 = pour IDX=1 : job id SLURM du dernier run_target terminé (ex: 61234567) — le
#        script retrouve son fichier de sortie et le filtre aux articles ayant au
#        moins une cible = True ; ou, alternativement, un chemin de fichier parquet
#        déjà filtré à utiliser tel quel.
#      pour IDX>1 : fichier d'entrée (fourni automatiquement par le chaînage interne).
IDX="${1:?Usage: sbatch sbatch_run_inst.sh <instrument_index 1-7> <target_job_id|input_file>}"
if [ "$IDX" -lt 1 ] || [ "$IDX" -gt "$N_INSTRUMENTS" ]; then
  echo "IDX must be between 1 and $N_INSTRUMENTS, got: $IDX" >&2
  exit 1
fi
CODE="${INSTRUMENTS[$((IDX - 1))]}"
CODE_LOWER=$(echo "$CODE" | tr '[:upper:]' '[:lower:]')

# Dernière cible de la chaîne run_target (voir TARGETS dans sbatch_run_target.sh) :
# son fichier de sortie contient les 12 colonnes target_<CODE> nécessaires au filtrage.
TARGET_LAST_CODE_LOWER="information_societal_harms"

OUTBASE="${OUTDIR}/laws_structure_with_instrument_${CODE_LOWER}"

cd "$WORKDIR"
source .venv/bin/activate

SECOND_ARG="${2:-}"

if [ "$IDX" -eq 1 ]; then
  if [ -z "$SECOND_ARG" ]; then
    echo "Pour l'instrument 1, fournir soit le job id du dernier run_target (ex: 61234567), soit un chemin de fichier parquet déjà filtré." >&2
    exit 1
  fi
  if [[ "$SECOND_ARG" =~ ^[0-9]+$ ]]; then
    TARGET_JOB_ID="$SECOND_ARG"
    TARGET_OUTPUT="${OUTDIR}/laws_structure_with_target_${TARGET_LAST_CODE_LOWER}_job${TARGET_JOB_ID}.parquet"
    if [ ! -f "$TARGET_OUTPUT" ]; then
      echo "Fichier de sortie run_target introuvable: $TARGET_OUTPUT (job id incorrect ?)" >&2
      exit 1
    fi
    INPUT="${OUTDIR}/laws_structure_target_relevant_job${TARGET_JOB_ID}.parquet"
    echo "Filtrage des articles avec >=1 cible = True depuis $TARGET_OUTPUT"
    python scripts/filter_target_relevant.py \
      --input "$TARGET_OUTPUT" \
      --output "$INPUT" \
      --level_col level
  else
    INPUT="$SECOND_ARG"
  fi
else
  INPUT="$SECOND_ARG"
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
echo "Instrument: $IDX/$N_INSTRUMENTS = $CODE"
echo "Input: $INPUT"
nvidia-smi -L || true

python scripts/run_inst_pipeline.py \
  --input  "$INPUT" \
  --output_base "$OUTBASE" \
  --instrument_code "$CODE" \
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

DECISION_COL="instrument_${CODE}"

# fichier benchmark gold (colonnes instrument_<CODE> attendues, une par instrument)
GOLD_CSV="data/external/PORTFOLIO_GOLD.csv"
GOLD_COL="Instrument_${CODE}"

TEMP_RUN_DIR="data/output/run_inst_${IDX}_${CODE}_job${SLURM_JOB_ID}"
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
  RUN_DIR="data/output/run_inst_${IDX}_${CODE}_no_score_job${SLURM_JOB_ID}"
else
  SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')
  RUN_DIR="data/output/run_inst_${IDX}_${CODE}_${SCORE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$RUN_DIR"

# --- Archive: outputs + prompt ---
cp "$PRED_CSV" "$RUN_DIR/" || true
cp "src/run_inst_prompts.py" "$RUN_DIR/prompts_used.py" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

if [ -d "$TEMP_RUN_DIR/eval" ]; then
  mv "$TEMP_RUN_DIR/eval" "$RUN_DIR/eval"
fi

rmdir "$TEMP_RUN_DIR" 2>/dev/null || true

echo "Archived in: $RUN_DIR"
echo "Score: ${SCORE}%"

# --- Chaînage : soumet automatiquement l'instrument suivant ---
# N'est atteint que si le run ci-dessus a réussi (set -e). Le fichier produit par ce
# stage devient l'entrée du stage suivant.
if [ "$IDX" -lt "$N_INSTRUMENTS" ]; then
  NEXT_IDX=$((IDX + 1))
  NEXT_CODE="${INSTRUMENTS[$((NEXT_IDX - 1))]}"
  echo "Chaining: submitting instrument $NEXT_IDX/$N_INSTRUMENTS = $NEXT_CODE"
  sbatch --job-name="run_inst_${NEXT_IDX}_${NEXT_CODE}" sbatch_run_inst.sh "$NEXT_IDX" "$PRED_PARQUET"
else
  echo "Dernier instrument ($CODE) traité — fin de la chaîne run_inst."
fi
