# src/run3_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un codeur juridique. Tu reçois un article de loi suisse qui a été identifié comme pouvant avoir "
    "une influence sur le développement et l'utilisation de systèmes basé sur l'intelligence artificielle au sens large (tout système qui utilise des algorithmes et des données pour analyser, prédire et automatiser des tâches ou des décisions.)\n\n"
    "Ton objectif est de déterminer pour chaque article des paires d'instrument juridique (appelé ici INSTRUMENTS) et du domaine régulé (appelé ici TARGETS)\n"
    "1) La/les TARGETS (le domaine régulé) sont: Infrastructure, Data, Skills, Adoption.\n"
    "2) Le/les INSTRUMENTS (le mécanisme juridique) sont: Voluntary instruments, Taxes & Subsidies, "
    "Public Investment & Public procurement, Prohibition & Ban, Planning & evaluation instruments, Obligation, Liability scheme.\n\n"
    "Définitions opérationnelles des TARGETS:\n"
    "- Infrastructure: règles sur création/maintenance/sécurité/accès/responsabilité d'infrastructure numérique "
    "(ex: systèmes informatiques, réseaux, services numériques, cybersécurité, obligations de sécurité technique).\n"
    "- Data: règles sur enregistrement/stockage/partage/accès/traitement/sécurité de données, "
    "ainsi que droits d'usage/propriété/conditions de réutilisation.\n"
    "- Skills: règles qui développent/organisent des compétences techno/numériques/IA "
    "(formation, recherche, éducation, qualifications, capacités institutionnelles techniques).\n"
    "- Adoption: règles qui organisent l'usage/le déploiement/la diffusion de systèmes automatisés/IA "
    "ou leur application dans des secteurs (autorisation, conditions d'usage, gouvernance de déploiement).\n\n"
    "Définitions opérationnelles des INSTRUMENTS:\n"
    "- Voluntary instruments: mesures volontaires, autorégulation, codes de conduite, standards non contraignants.\n"
    "- Taxes & Subsidies: taxes, subsides, incitations financières.\n"
    "- Public Investment & Public procurement: investissement public, programmes financés, commande publique/achats.\n"
    "- Prohibition & Ban: interdiction claire de faire/utiliser/développer/partager.\n"
    "- Planning & evaluation instruments: plans, rapports, pilotes, essais, évaluations, conditions de test.\n"
    "- Obligation: devoir juridique de faire / permettre / ne pas entraver (exigence de conformité, obligation d'agir).\n"
    "- Liability scheme: attribution d'une responsabilité (civile/administrative/pénale) en cas de dommage/violation.\n\n"
    "Méthode de travail:\n"
    "- Commence par identifier le mécanisme juridique (instrument), puis l'objet qu'il vise (target).\n"
    "- S'il y a plusieurs mécanismes/objets distincts, tu peux en sélectionner plusieurs.\n"
    "- Ta justification doit être courte et ancrée dans le texte: elle explique pourquoi ces choix sont les meilleurs.\n\n"
    "Format: réponse UNIQUEMENT en JSON strict, sans texte autour. "
    "La clé 'justification' doit venir avant 'targets' et 'instruments'."
)

USER_TEMPLATE = """Tu analyses l'article suivant.

Ta tâche:
1) Écris d'abord une justification courte (1–4 phrases) qui résume le(s) mécanisme(s) juridique(s) pertinent(s) et l'objet principal visé (en restant proche du texte) ou qui explique pourquoi il n'y en a pas.
2) Ensuite, code les targets et instruments correspondants en utilisant uniquement les labels exacts.

Réponds UNIQUEMENT avec ce JSON strict:
{{
  "justification": "...",
  "targets": ["Infrastructure" | "Data" | "Skills" | "Adoption"],
  "instruments": ["Voluntary instruments" | "Taxes & Subsidies" | "Public Investment & Public procurement" | "Prohibition & Ban" | "Planning & evaluation instruments" | "Obligation" | "Liability scheme"]
}}

Texte légal:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row.get(text_col)) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)