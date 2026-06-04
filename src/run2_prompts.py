# src/run2_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un auditeur critique chargé de valider des classifications automatiques.\n\n"
    "Un modèle de langage a analysé des articles de loi pour déterminer si chacun "
    "contient un instrument de politique publique (OUI) ou non (NON). "
    "Un instrument de politique publique est un mécanisme concret par lequel l'État "
    "impose, interdit, incite, finance ou sanctionne un comportement d'acteurs "
    "externes (individus, entreprises, organisations).\n\n"
    "Ta tâche est de vérifier si la classification est correcte en analysant "
    "l'article avec un regard critique, en cherchant aussi bien les faux positifs "
    "que les faux négatifs.\n\n"
    "Une classification OUI est un FAUX POSITIF si l'article :\n"
    "- définit uniquement des termes ou concepts sans créer d'obligation ou "
    "d'interdiction sur des acteurs extérieurs\n"
    "- organise le fonctionnement interne d'une autorité (ses compétences, "
    "ses procédures internes) sans affecter des acteurs extérieurs\n"
    "- délègue facultativement un pouvoir réglementaire (« peut prévoir », "
    "« peut décider ») sans préciser les obligations concrètes imposées aux "
    "acteurs extérieurs dans l'article lui-même\n"
    "- décrit les pouvoirs de contrôle ou de surveillance d'un organe public "
    "sans créer de nouvelles obligations concrètes pour les entités surveillées\n\n"
    "Une classification NON est un FAUX NÉGATIF si l'article contient :\n"
    "- une obligation directe sur des acteurs externes (« doit », « est tenu de », "
    "futur à valeur impérative)\n"
    "- une interdiction directe (« il est interdit de », « ne peut pas »)\n"
    "- un financement, une subvention ou une taxe visant des acteurs extérieurs\n"
    "- une condition légale pour obtenir ou conserver une autorisation, un permis "
    "ou une licence\n"
    "- une sanction, une responsabilité civile ou une assurance obligatoire\n\n"
    "Réponds en deux parties dans cet ordre exact :\n"
    "Audit: [1 à 2 phrases justifiant ta décision]\n"
    "Décision: CONFIRME ou INFIRME"
)

USER_TEMPLATE = """Article :
{article_text}

Classification produite par le modèle : {instrument_decision}

Cette classification est-elle correcte ?"""


def build_user_prompt(row: pd.Series, text_col: str, instrument_col: str = "instrument") -> str:
    txt = "" if pd.isna(row.get(text_col)) else str(row[text_col]).strip()
    val = row.get(instrument_col)
    if pd.isna(val):
        decision = "INDÉTERMINÉE"
    else:
        decision = "OUI (contient un instrument)" if bool(val) else "NON (ne contient pas d'instrument)"
    return USER_TEMPLATE.format(article_text=txt, instrument_decision=decision)
