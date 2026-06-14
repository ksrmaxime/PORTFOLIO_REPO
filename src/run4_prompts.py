# src/run4_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un auditeur critique chargé de valider des classifications automatiques.\n\n"
    "Un modèle de langage a analysé des articles de loi pour déterminer si chacun est "
    "lié à l'intelligence artificielle (systèmes automatisés, algorithmes, décisions "
    "sans intervention humaine). Il a produit une décision (OUI ou NON) et une justification.\n\n"
    "Ta tâche est de vérifier que la décision est correcte en lisant l'article toi-même.\n\n"
    "RÈGLE ABSOLUE : chaque affirmation dans la justification doit être directement "
    "vérifiable dans le texte de l'article. Une référence à l'IA, aux algorithmes ou "
    "aux systèmes automatisés qui n'apparaît pas dans l'article est une hallucination.\n\n"
    "Si la décision est OUI :\n"
    "- INFIRME si l'article ne contient aucune mention explicite de systèmes automatisés, "
    "d'algorithmes ou de décisions sans intervention humaine\n"
    "- INFIRME si la justification confond « numérique » ou « électronique » avec « automatisé »\n"
    "- INFIRME si la justification se fonde sur un lien hypothétique "
    "(« pourrait utiliser l'IA », « compatible avec l'IA »)\n"
    "- CONFIRME uniquement si le texte lui-même vise explicitement des systèmes automatisés\n\n"
    "Si la décision est NON :\n"
    "- INFIRME si l'article contient en réalité des références explicites à des systèmes "
    "automatisés, des algorithmes ou des décisions sans intervention humaine que la "
    "justification a ignorées\n"
    "- CONFIRME si l'article traite effectivement d'un autre sujet sans lien avec l'IA\n\n"
    "Réponds en deux parties dans cet ordre exact :\n"
    "Audit: [1 à 2 phrases ancrées dans le texte de l'article]\n"
    "Décision: CONFIRME ou INFIRME"
)

USER_TEMPLATE = """Article :
{article_text}

Décision du classificateur : {ai_relevant_decision}
Justification produite par le classificateur :
{run3_justif}

Cette décision est-elle correcte ?"""


def build_user_prompt(
    row: pd.Series,
    text_col: str,
    justif_col: str = "RUN3_JUSTIF",
    ai_relevant_col: str = "AI_RELEVANT",
) -> str:
    txt = "" if pd.isna(row.get(text_col)) else str(row[text_col]).strip()
    justif = "" if pd.isna(row.get(justif_col)) else str(row[justif_col]).strip()
    val = row.get(ai_relevant_col)
    decision = "OUI" if val is True or val == True else "NON"
    return USER_TEMPLATE.format(
        article_text=txt,
        run3_justif=justif,
        ai_relevant_decision=decision,
    )
