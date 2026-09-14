# src/run6_config.py
from __future__ import annotations

from collections import OrderedDict

from src.run5_prompts import INSTRUMENT_CODES
from src.run_target_prompts import TARGET_DEFINITIONS

# ---------------------------------------------------------------------------
# Targets: sourced from src/run_target_prompts.TARGET_DEFINITIONS, the module
# that defines the actual 10-target taxonomy used by the run_target_* LLM
# chain (sbatch_run_target.sh) and by PORTFOLIO_GOLD.csv (columns
# Target_<CODE>). This taxonomy replaces the older, unrelated 10-target
# taxonomy in src/run5_prompts.py (PERSONAL_DATA, IP_CREATIVE_CONTENT,
# EDUCATION, RESEARCH, COMPUTE_HARDWARE, DATA_CENTERS_ENERGY,
# HIGH_STAKES_APPS, ALGORITHMIC_ACCOUNTABILITY, DISINFORMATION,
# CYBERSECURITY_AI) — run5_prompts.TARGET_CODES must NOT be used here.
#
# TARGET_DEFINITIONS is already ordered by the 4 quadrants of the PoC PDF
# (Fig. 1 / section 3.1): Enabling x Upstream (4), Safeguarding x Upstream
# (2), Enabling x Downstream (1), Safeguarding x Downstream (3). We reuse
# that order and its "quadrant" field directly instead of re-declaring a
# separate grouping that could drift out of sync.
# ---------------------------------------------------------------------------

TARGET_CODES: "OrderedDict[str, str]" = OrderedDict(
    (code, d["name"]) for code, d in TARGET_DEFINITIONS.items()
)
TARGET_ORDER: list[str] = list(TARGET_CODES.keys())

# English display names matching the PoC PDF verbatim (Table in section
# 4.2.2 / Figure 1) — distinct from TARGET_CODES above, whose French names
# are only the internal coding labels used by the (French-language) LLM
# prompts in run_target_prompts.py. The final portfolio figure should read
# like Figure 1, regardless of the corpus/prompt language.
TARGET_DISPLAY_NAMES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("RESEARCH_INNOVATION", "Research & Innovation"),
        ("SKILLS_HUMAN_CAPITAL", "Skills & Human Capital"),
        ("DATA_ACCESS_RESOURCES", "Data Access & Resources"),
        ("COMPUTE_INFRASTRUCTURE", "Compute & Infrastructure"),
        ("DATA_PRIVACY_IP", "Data, Privacy & Intellectual Property"),
        ("SECURITY_ROBUSTNESS", "Security & Robustness"),
        ("AI_DEPLOYMENT", "AI Deployment"),
        ("ACCOUNTABILITY_TRANSPARENCY", "System Accountability & Transparency"),
        ("OUTPUT_HARMS", "Output-Related Harms"),
        ("SOCIETAL_HARMS", "Societal Harms"),
    ]
)
if set(TARGET_DISPLAY_NAMES) != set(TARGET_ORDER):
    raise RuntimeError("TARGET_DISPLAY_NAMES in run6_config.py is out of sync with TARGET_DEFINITIONS")

# code -> ("Enabling"|"Safeguarding", "Upstream"|"Downstream")
TARGET_QUADRANT: "OrderedDict[str, tuple[str, str]]" = OrderedDict(
    (code, tuple(part.strip() for part in d["quadrant"].split("x")))
    for code, d in TARGET_DEFINITIONS.items()
)


def _contiguous_runs(keys: list[str]) -> list[tuple[str, int, int]]:
    """Collapse a list of per-column keys into (key, start, end) runs of
    consecutive columns sharing the same key, preserving column order."""
    runs: list[tuple[str, int, int]] = []
    for i, key in enumerate(keys):
        if runs and runs[-1][0] == key:
            prev_key, start, _ = runs[-1]
            runs[-1] = (prev_key, start, i + 1)
        else:
            runs.append((key, i, i + 1))
    return runs


# Contiguous Upstream/Downstream runs (the outer band in Figure 1), e.g.
# [("Upstream", 0, 6), ("Downstream", 6, 10)].
TARGET_LOCATION_RUNS: list[tuple[str, int, int]] = _contiguous_runs(
    [location for _, location in TARGET_QUADRANT.values()]
)

# Contiguous Enabling/Safeguarding runs (the inner band in Figure 1), e.g.
# [("Enabling", 0, 4), ("Safeguarding", 4, 6), ("Enabling", 6, 7),
#  ("Safeguarding", 7, 10)].
TARGET_FUNCTION_RUNS: list[tuple[str, int, int]] = _contiguous_runs(
    [function for function, _ in TARGET_QUADRANT.values()]
)

# ---------------------------------------------------------------------------
# Instruments: the 7-category taxonomy is unchanged between run5_prompts.py
# and run_inst_prompts.py, so src/run5_prompts.INSTRUMENT_CODES remains the
# single source of truth here.
# ---------------------------------------------------------------------------

INSTRUMENT_ORDER: list[str] = list(INSTRUMENT_CODES.keys())
