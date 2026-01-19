# PORTFOLIO_REPO

Workspace for the Switzerland AI Regulation Portfolio (LLM-impact scope).

## Setup
1) Create venv (Python 3.12.12):
   - python3.12 -m venv .venv
2) Activate:
   - source .venv/bin/activate
3) Install deps (once pyproject is defined):
   - python -m pip install -U pip
   - pip install -e ".[dev]"

## Structure
- data/        raw inputs (not committed if sensitive)
- notebooks/   exploration
- outputs/     generated outputs (figures, exports)
- scripts/     runnable scripts/CLI entry points
- src/         package code
