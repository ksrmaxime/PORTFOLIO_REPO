# OLD — pipeline exploratoire (archivée le 2026-10-15)

Codes de la première phase d'exploration (run0 → run6, run_target, run_target_control,
run_inst, scoring). Conservés pour référence uniquement : ils ne font plus partie de la
pipeline active et leurs imports (`from src.run5_prompts ...`, `ROOT = parents[1]`)
peuvent ne plus fonctionner depuis le déplacement.

- `OLD/sbatch/`     : scripts SLURM des anciens runs
- `OLD/logs/`       : logs SLURM des anciens runs
- `OLD/notebooks/`  : exploration
- `src/OLD/`        : prompts, configs, plot et scoring des anciens runs
- `scripts/OLD/`    : pipelines des anciens runs + utilitaires (score, add_row_id, PoC_filter...)

Restent actifs : `src/download_1.py` + `scripts/download.py` (téléchargement Fedlex et
découpe article par article), `src/client.py` et `src/runner.py` (exécution LLM sur le cluster).
