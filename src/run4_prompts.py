# src/run4_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un auditeur critique chargé de valider des classifications automatiques.\n\n"
    "Un modèle de langage a analysé des articles de loi pour déterminer si chacun "
    "contient un instrument de politique publique ciblant un problème public lié à "
    "l'intelligence artificielle. Il a produit une décision (OUI) et une justification.\n\n"
    "RÈGLE ABSOLUE : tu dois lire l'article toi-même et vérifier que chaque affirmation "
    "de la justification est directement vérifiable dans le texte. Une justification qui "
    "mentionne l'IA, les algorithmes ou les systèmes automatisés sans que ces éléments "
    "soient présents ou clairement impliqués dans l'article est une hallucination — "
    "elle doit être rejetée même si elle semble bien rédigée.\n\n"
    "Ton rôle est de détecter les faux positifs : les cas où le modèle a conclu OUI "
    "mais s'est montré trop accommodant dans son raisonnement.\n\n"
    "Une classification OUI est un FAUX POSITIF si :\n"
    "- l'article ne contient aucune mention explicite de systèmes automatisés, "
    "d'algorithmes, de décisions sans intervention humaine ou de traitement algorithmique "
    "— quelle que soit la qualité apparente de la justification\n"
    "- la justification cite un secteur sensible (santé, justice, sécurité) sans que "
    "l'article cible spécifiquement des systèmes automatisés\n"
    "- la justification confond « numérique » ou « électronique » avec « automatisé »\n"
    "- la justification se fonde sur un lien hypothétique "
    "(« ce domaine pourrait utiliser l'IA », « cela est compatible avec l'IA »)\n\n"
    "Une classification OUI est CORRECTE uniquement si le texte de l'article lui-même "
    "contient des termes ou mécanismes visant explicitement des systèmes automatisés, "
    "des algorithmes ou des décisions prises sans intervention humaine.\n\n"
    "Réponds en deux parties dans cet ordre exact :\n"
    "Audit: [1 à 2 phrases ancrées dans le texte de l'article, pas dans la justification]\n"
    "Décision: CONFIRME ou INFIRME"
)

USER_TEMPLATE = """Article :
{article_text}

Justification produite par le classificateur :
{run3_justif}

Cette justification est-elle solide, ou le classificateur a-t-il été chercher trop loin ?"""


def build_user_prompt(row: pd.Series, text_col: str, justif_col: str = "RUN3_JUSTIF") -> str:
    txt = "" if pd.isna(row.get(text_col)) else str(row[text_col]).strip()
    justif = "" if pd.isna(row.get(justif_col)) else str(row[justif_col]).strip()
    return USER_TEMPLATE.format(article_text=txt, run3_justif=justif)
