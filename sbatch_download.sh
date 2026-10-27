#!/bin/bash -l
#SBATCH --job-name=fedlex_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fedlex_download_%j.out
#SBATCH --error=logs/fedlex_download_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Téléchargement Fedlex, FR, et découpe article par article -> laws_structure_final.parquet / .csv
#   - Recueil systématique (RS) : actes en vigueur à AS_OF ;
#   - Feuille fédérale (FF) : arrêtés fédéraux (tous types) publiés en XML (depuis mi-2021),
#     jusqu'à AS_OF.
#
# Usage :
#   sbatch sbatch_download.sh                      # tout (défaut) : RS (droit interne + traités) + arrêtés fédéraux
#   sbatch sbatch_download.sh arretes              # arrêtés fédéraux de la FF seulement (test rapide, ~380 actes)
#   sbatch sbatch_download.sh interne              # droit interne seulement
#   sbatch sbatch_download.sh all 20260101         # tout, en vigueur au 1.1.2026
#
# Groupes : interne (lois, ordonnances, autres actes fédéraux, conventions Confédération-
# cantons, arrêtés fédéraux) | traites | arretes | all | lois | ordonnances | autres_actes | conventions_cantons
#   REUSE=1 sbatch sbatch_download.sh              # réutiliser les XML déjà présents (pas de re-téléchargement)
#
# Par défaut, tous les XML sont re-téléchargés (raw_xml/ est écrasé).
# Prérequis : le nœud de calcul doit pouvoir joindre https://fedlex.data.admin.ch
# (vérifié en début de job).

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

DOC_TYPES="${1:-all}"
AS_OF="${2:-$(date +%Y%m%d)}"
REUSE="${REUSE:-0}"

case "$DOC_TYPES" in
  interne|traites|arretes|all|lois|ordonnances|autres_actes|conventions_cantons) ;;
  *) echo "DOC_TYPES invalide (reçu : $DOC_TYPES) — voir l'en-tête du script" >&2; exit 1 ;;
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
echo "DOC_TYPES=$DOC_TYPES AS_OF=$AS_OF REUSE=$REUSE"
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
# plafond mémoire Python = 85 % de la mémoire SLURM : un acte pathologique est ignoré
# (et signalé dans le log) au lieu de faire tuer le job ; chaque acte est logué avant découpe.
MEM_MB="${SLURM_MEM_PER_NODE:-65536}"
MAX_MEM_GB=$(( MEM_MB * 85 / 100 / 1024 ))
EXTRA_ARGS=(--verbose --max-mem-gb "$MAX_MEM_GB")
if [ "$REUSE" != "1" ]; then
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

# Versions CSV (lisibles dans Excel / Numbers) à côté de chaque parquet.
# utf-8-sig = UTF-8 avec BOM, pour que les accents s'affichent correctement dans Excel.
df.to_csv(f"{out}/laws_structure_final.csv", index=False, encoding="utf-8-sig")
pd.read_parquet(f"{out}/cc_catalog_fr_inforce.parquet").to_csv(f"{out}/cc_catalog_fr_inforce.csv", index=False, encoding="utf-8-sig")
log.to_csv(f"{out}/download_log.csv", index=False, encoding="utf-8-sig")
arts = df[df["level"] == 6]
print("=== BILAN ===")
print(f"XML ok: {int(log['ok'].sum()):,} / {len(log):,}  (dont déjà présents: {int(log['skipped_existing'].sum()):,})")
fails = log[~log["ok"].astype(bool)]
if len(fails):
    print(f"Échecs ({len(fails)}) :")
    print(fails[["base_act_uri", "error"]].head(20).to_string(index=False))
print("Actes par type :")
print(df.drop_duplicates("law_id")["doc_type_label"].value_counts().to_string())
cat = pd.read_parquet(f"{out}/cc_catalog_fr_inforce.parquet")
print("Base vigueur :", cat["in_force_basis"].value_counts().to_dict())
print("Actes par catégorie :", df.drop_duplicates("law_id")["doc_category"].value_counts().to_dict())
print(f"Articles : {len(arts):,} (dont sans texte / abrogés : {int((arts['text'].fillna('').str.strip() == '').sum()):,} ; actes sans articles, texte intégral en 1 ligne : {int((arts['label'] == 'Texte intégral (sans articles)').sum()):,})")
PY

echo "Terminé :"
echo "  $OUTDIR/laws_structure_final.parquet"
echo "  $OUTDIR/laws_structure_final.csv"
echo "  (+ cc_catalog_fr_inforce.parquet/.csv, download_log.parquet/.csv)"
