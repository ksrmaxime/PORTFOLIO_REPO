# src/prompts.py (portfolio run2)
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un système STRICT de classification juridique.\n"
    "Tu dois répondre exactement sur DEUX lignes, dans cet ordre, sans texte supplémentaire:\n"
    "RELEVANT: TRUE ou FALSE\n"
    "JUSTIFICATION: une justification courte (1-2 phrases) en français\n"
    "Ne mets rien d'autre."
)

USER_TEMPLATE = """Décide si cet article fixe des règles/obligations/procédures/responsabilités/exigences
qui influence l’implémentation, le développement, la sécurité, l’usage ou la responsabilité
d’une intelligence artificielle (au sens large) : système qui utilise des algorithmes et des données pour analyser, prédire et automatiser des tâches ou des décisions.

Important:
- TRUE seulement si l'article aborde une dimension de l'intelligence artificielle et s’il y a un effet normatif concret lié à ces systèmes (conditions, exigences, surveillance, obligations, responsabilités, procédure, autorisations, etc.).
- FALSE si aucune forme d'intelligence artificielle n'est impliqée ou que c’est seulement descriptif/définition générale ou sans portée normative liée à ces systèmes.

Texte légal:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)