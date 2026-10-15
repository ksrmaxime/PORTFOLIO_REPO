#!/bin/bash -l
#SBATCH --job-name=fedlex_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fedlex_download_%j.out
#SBATCH --error=logs/fedlex_download_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Téléchargement Fedlex (lois + ordonnances fédérales en vigueur, FR) et découpe
# article par article -> laws_structure_final.parquet / .csv
#
# Usage :
#   sbatch sbatch_download.sh                      # lois + ordonnances, version en vigueur aujourd'hui
#   sbatch sbatch_download.sh lois                 # lois seulement   (lois | ordonnances | all)
#   sbatch sbatch_download.sh all 20260101         # version consolidée en vigueur au 1.1.2026
#   OVERWRITE=1 sbatch sbatch_download.sh          # re-télécharger les XML déjà présents
#
# Les XML déjà présents dans $OUTDIR/raw_xml sont réutilisés (cache), donc relancer
# le job après un échec ne retélécharge que ce qui manque.
# Prérequis : le nœud de calcul doit pouvoir joindre https://fedlex.data.admin.ch
# (vérifié en début de job).

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

DOC_TYPES="${1:-all}"
AS_OF="${2:-$(date +%Y%m%d)}"
OVERWRITE="${OVERWRITE:-0}"

case "$DOC_TYPES" in
  lois|ordonnances|all) ;;
  *) echo "DOC_TYPES doit être lois, ordonnances ou all (reçu : $DOC_TYPES)" >&2; exit 1 ;;
esac
if ! [[ "$AS_OF" =~ ^[0-9]{8}$ ]]; then
  echo "AS_OF doit être au format YYYYMMDD (reçu : $AS_OF)" >&2; exit 1
fi

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/PORTFOLIO_REPO
OUTDIR="${WORKDIR}/data/processed/fedlex/${DOC_TYPES}"

cd "$WORKDIR"
source .venv/bin/activate
export PYTHONPATH="${WORKDIR}/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "DATE=$(date -Is)"
echo "DOC_TYPES=$DOC_TYPES AS_OF=$AS_OF OVERWRITE=$OVERWRITE"
echo "OUTDIR=$OUTDIR"
echo "GIT=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
python -V

# --- Vérifications préalables (échouer vite plutôt qu'après 4h) ---
python -c "import pandas, pyarrow, requests, lxml; print('deps OK')"
HTTP_CODE=$(curl -sS -m 30 -o /dev/null -w "%{http_code}" \
  -H "Accept: application/sparql-results+json" \
  --data-urlencode "query=ASK {}" \
  https://fedlex.data.admin.ch/sparqlendpoint || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
  echo "Fedlex injoignable depuis $(hostname) (HTTP $HTTP_CODE). Le nœud a-t-il accès à internet ?" >&2
  exit 1
fi
echo "Fedlex OK (HTTP $HTTP_CODE)"

# --- Téléchargement + découpe ---
EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ]; then
  EXTRA_ARGS+=(--overwrite-download)
fi

python scripts/download.py \
  --out-dir "$OUTDIR" \
  --doc-types "$DOC_TYPES" \
  --as-of "$AS_OF" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# --- Bilan ---
python - <<PY
import pandas as pd
out = "$OUTDIR"
log = pd.read_parquet(f"{out}/download_log.parquet")
df = pd.read_parquet(f"{out}/laws_structure_final.parquet")
arts = df[df["level"] == 6]
print("=== BILAN ===")
print(f"XML ok: {int(log['ok'].sum()):,} / {len(log):,}  (dont déjà présents: {int(log['skipped_existing'].sum()):,})")
fails = log[~log["ok"].astype(bool)]
if len(fails):
    print(f"Échecs ({len(fails)}) :")
    print(fails[["base_act_uri", "error"]].head(20).to_string(index=False))
print("Actes par type :")
print(df.drop_duplicates("law_id")["doc_type_label"].value_counts().to_string())
print(f"Articles : {len(arts):,} (dont sans texte / abrogés : {int((arts['text'].fillna('').str.strip() == '').sum()):,})")
PY

echo "Terminé : $OUTDIR/laws_structure_final.parquet"
