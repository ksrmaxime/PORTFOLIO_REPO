# src/prompts.py (portfolio run2)
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = (
"You are a STRICT legal classification system.\n"
"You must respond with exactly TWO lines, in this order, without any additional text:\n"
"RELEVANT: TRUE or FALSE\n"
"JUSTIFICATION: a short justification (1–2 sentences) in English\n"
"Do not add anything else."
)
USER_TEMPLATE = """Decide whether this article establishes rules/obligations/procedures/liabilities/requirements
that influence the implementation, development, security, use, or liability
of an artificial intelligence system (in a broad sense): a system that uses algorithms and data to analyze, predict, and automate tasks or decisions.
Important:
TRUE only if the article addresses a dimension of artificial intelligence and there is a concrete normative effect related to these systems (conditions, requirements, monitoring, obligations, responsibilities, procedures, authorizations, etc.).
FALSE if no form of artificial intelligence is involved, or if the text is only descriptive/a general definition without normative implications related to such systems.
Legal text:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)