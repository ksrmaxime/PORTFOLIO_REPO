# src/run2_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un auditeur critique chargé de valider des classifications automatiques.\n\n"
    "Un modèle de langage a analysé des articles de loi pour déterminer si chacun "
    "contient un instrument de politique publique (OUI) ou non (NON), et a fourni "
    "une justification de sa décision.\n\n"
    "Ta tâche est de vérifier si la justification est solide et si la décision "
    "qui en découle est correcte.\n\n"
    "Une classification OUI est un FAUX POSITIF si la justification :\n"
    "- confond une définition ou un champ d'application avec une obligation "
    "imposée à des acteurs extérieurs\n"
    "- attribue un instrument à une disposition qui organise uniquement le "
    "fonctionnement interne d'une autorité\n"
    "- cite une délégation facultative de pouvoir (« peut prévoir », « peut décider ») "
    "comme si elle créait des obligations concrètes imposées à des acteurs extérieurs\n"
    "- assimile des pouvoirs de contrôle ou de surveillance à des obligations "
    "nouvelles pour les entités surveillées\n\n"
    "Une classification NON est un FAUX NÉGATIF si la justification :\n"
    "- ignore une obligation directe explicite dans le texte "
    "(« doit », « est tenu de », futur à valeur impérative)\n"
    "- ignore une interdiction directe (« il est interdit de », « ne peut pas »)\n"
    "- ignore une sanction, une responsabilité civile ou une assurance obligatoire\n"
    "- ignore des conditions légales pour obtenir ou conserver une autorisation, "
    "un permis ou une licence\n\n"
    "Une justification est CORRECTE si elle identifie avec précision le mécanisme "
    "concret présent ou absent dans le texte qui fonde la décision OUI ou NON.\n\n"
    "Réponds en deux parties dans cet ordre exact :\n"
    "Audit: [1 à 2 phrases évaluant la solidité de la justification]\n"
    "Décision: CONFIRME ou INFIRME"
)

USER_TEMPLATE = """Article :
{article_text}

Classification produite par le modèle : {instrument_decision}
Justification : {run1_justif}

Cette justification est-elle solide et la classification correcte ?"""


def build_user_prompt(
    row: pd.Series,
    text_col: str,
    instrument_col: str = "instrument",
    justif_col: str = "RUN1_JUSTIF",
) -> str:
    txt = "" if pd.isna(row.get(text_col)) else str(row[text_col]).strip()
    val = row.get(instrument_col)
    if pd.isna(val):
        decision = "INDÉTERMINÉE"
    else:
        decision = "OUI (contient un instrument)" if bool(val) else "NON (ne contient pas d'instrument)"
    justif = "" if pd.isna(row.get(justif_col)) else str(row[justif_col]).strip()
    return USER_TEMPLATE.format(article_text=txt, instrument_decision=decision, run1_justif=justif)
