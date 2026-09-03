#!/bin/bash -l
#SBATCH --job-name=run_target_control
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_target_control_%j.out
#SBATCH --error=logs/run_target_control_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO/data/processed

# Cette chaîne tourne APRÈS que la chaîne run_target (sbatch_run_target.sh, 10
# étapes) est arrivée à son terme. Elle reprend, cible par cible, uniquement
# les articles classés OUI par run_target pour cette cible (colonne
# target_<CODE> == True) et redemande au LLM de confirmer cette classification
# avec une définition très succincte de la cible (au lieu de la définition
# détaillée du premier passage). Résultat écrit dans control_target_<CODE>.
#
# --- Doit rester synchronisé (mêmes codes, même ordre) avec TARGETS dans
# sbatch_run_target.sh et TARGET_DEFINITIONS dans src/run_target_prompts.py.
TARGETS=(RESEARCH_INNOVATION SKILLS_HUMAN_CAPITAL DATA_ACCESS_RESOURCES COMPUTE_INFRASTRUCTURE DATA_PRIVACY_IP SECURITY_ROBUSTNESS AI_DEPLOYMENT ACCOUNTABILITY_TRANSPARENCY OUTPUT_HARMS SOCIETAL_HARMS)
N_TARGETS=${#TARGETS[@]}

# Dernière cible de la chaîne run_target : son fichier de sortie contient les
# 10 colonnes target_<CODE> nécessaires au contrôle.
TARGET_LAST_CODE_LOWER="societal_harms"

# $1 = index de cible (1-10)
# $2 = pour IDX=1 : job id SLURM de la dernière étape run_target terminée (ex:
#        61234567), ou alternativement un chemin de fichier parquet déjà
#        produit par la chaîne run_target à utiliser tel quel ;
#      pour IDX>1 : fichier d'entrée (fourni automatiquement par le chaînage
#        interne).
IDX="${1:?Usage: sbatch sbatch_run_target_control.sh <target_index 1-10> <run_target_job_id|input_file>}"
if [ "$IDX" -lt 1 ] || [ "$IDX" -gt "$N_TARGETS" ]; then
  echo "IDX must be between 1 and $N_TARGETS, got: $IDX" >&2
  exit 1
fi
CODE="${TARGETS[$((IDX - 1))]}"
CODE_LOWER=$(echo "$CODE" | tr '[:upper:]' '[:lower:]')

OUTBASE="${OUTDIR}/laws_structure_with_target_control_${CODE_LOWER}"

cd "$WORKDIR"
source .venv/bin/activate

SECOND_ARG="${2:-}"

if [ "$IDX" -eq 1 ]; then
  if [ -z "$SECOND_ARG" ]; then
    echo "Pour la cible 1, fournir soit le job id de la dernière étape run_target (ex: 61234567), soit un chemin de fichier parquet déjà produit par la chaîne run_target." >&2
    exit 1
  fi
  if [[ "$SECOND_ARG" =~ ^[0-9]+$ ]]; then
    TARGET_JOB_ID="$SECOND_ARG"
    INPUT="${OUTDIR}/laws_structure_with_target_${TARGET_LAST_CODE_LOWER}_job${TARGET_JOB_ID}.parquet"
    if [ ! -f "$INPUT" ]; then
      echo "Fichier de sortie run_target introuvable: $INPUT (job id incorrect ?)" >&2
      exit 1
    fi
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
echo "Control target: $IDX/$N_TARGETS = $CODE"
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
  --batch_size 8 \
  --max_new_tokens 150 \
  --temperature 0.0

# --- Bilan : taux de confirmation (pas de comparaison à un gold, il s'agit
# d'un second avis sur les propres classifications de run_target) ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"
PRED_PARQUET="${OUTBASE}_job${SLURM_JOB_ID}.parquet"

SOURCE_COL="target_${CODE}"
DECISION_COL="control_target_${CODE}"

CONFIRM_LOG=$(python -c "
import pandas as pd
df = pd.read_csv('$PRED_CSV')
mask = df['$SOURCE_COL'].fillna(False).astype(bool)
sub = df.loc[mask, '$DECISION_COL']
n = len(sub)
n_confirmed = int((sub == True).sum())
n_infirmed = int((sub == False).sum())
n_na = int(sub.isna().sum())
print(f'n_sent={n} n_confirmed={n_confirmed} n_infirmed={n_infirmed} n_na={n_na}')
print(f'ConfirmRate: {100.0 * n_confirmed / n:.2f}%' if n else 'ConfirmRate: NA')
")

echo "$CONFIRM_LOG"

CONFIRM_RATE=$(echo "$CONFIRM_LOG" | awk -F': ' '/^ConfirmRate:/ {gsub(/%/,"",$2); print $2; exit}')
CONFIRM_RATE=${CONFIRM_RATE:-NA}

if [ "$CONFIRM_RATE" = "NA" ]; then
  RUN_DIR="data/output/run_target_control_${IDX}_${CODE}_no_rate_job${SLURM_JOB_ID}"
else
  RATE_TAG=$(printf "%.2f" "$CONFIRM_RATE" | tr '.' 'p')
  RUN_DIR="data/output/run_target_control_${IDX}_${CODE}_${RATE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$RUN_DIR"

# --- Archive: outputs + prompt ---
cp "$PRED_CSV" "$RUN_DIR/" || true
cp "src/run_target_control_prompts.py" "$RUN_DIR/prompts_used.py" || true
cp "$0" "$RUN_DIR/sbatch_used.sbatch" || true

echo "Archived in: $RUN_DIR"
echo "Confirm rate: ${CONFIRM_RATE}%"

# --- Chaînage : soumet automatiquement la cible de contrôle suivante ---
# N'est atteint que si le run ci-dessus a réussi (set -e). Le fichier produit
# par ce stage devient l'entrée du stage suivant.
if [ "$IDX" -lt "$N_TARGETS" ]; then
  NEXT_IDX=$((IDX + 1))
  NEXT_CODE="${TARGETS[$((NEXT_IDX - 1))]}"
  echo "Chaining: submitting control target $NEXT_IDX/$N_TARGETS = $NEXT_CODE"
  sbatch --job-name="run_target_control_${NEXT_IDX}_${NEXT_CODE}" sbatch_run_target_control.sh "$NEXT_IDX" "$PRED_PARQUET"
else
  echo "Dernière cible ($CODE) contrôlée — fin de la chaîne run_target_control."
fi
