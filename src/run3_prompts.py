# src/run3_prompts.py
from __future__ import annotations

import pandas as pd

# IMPORTANT:
# - On veut "justification" AVANT les décisions (targets/instruments).
# - Sortie strictement en JSON (plus robuste à parser que du texte libre).

SYSTEM_PROMPT = (
    "Tu es un système STRICT de codage juridique pour un projet 'AI Regulation Portfolio' (Suisse).\n"
    "Tu analyses UN SEUL article (texte légal) à la fois.\n\n"
    "Objectif: extraire (i) une justification courte et (ii) les TARGETS AI-relevant et (iii) les INSTRUMENTS.\n\n"
    "RÈGLE MÉTHODOLOGIQUE CRITIQUE:\n"
    "- Préfère les faux négatifs aux faux positifs.\n"
    "- Un article n'est AI-relevant que s'il régule un objet qui conditionne directement des systèmes d'IA (notamment LLMs).\n"
    "- Ne déduis PAS la pertinence IA du secteur ou du contexte; uniquement de l'objet régulé.\n\n"
    "TARGETS possibles (liste fermée; 0..n):\n"
    "- Data (training/fine-tuning/evaluation; traitement automatisé; réutilisation/agrégation/inférence automatisées)\n"
    "- Computing Infrastructure (serveurs/cloud/stockage/sécurité/intégrité/accès/compute qui conditionne l'exécution)\n"
    "- Development & Adoption (autorise/restreint/gouverne des systèmes automatisés; décision automatisée; human-in-the-loop)\n"
    "- Skills (exigences/formation/qualifications explicitement liées à capacités AI/data/automation)\n\n"
    "EXCLUSIONS (si c'est seulement ça -> aucun target):\n"
    "- collecte/analyse de données uniquement pour un but métier (environnement/santé/traffic/statistique/reporting) sans contrainte de traitement automatisé.\n"
    "- digitalisation procédurale (e-filing, e-signature) sans impact sur compute/algorithmes.\n"
    "- 'state of the art' non-computationnel.\n\n"
    "INSTRUMENTS possibles (liste fermée; 0..n):\n"
    "- Voluntary instrument (autorégulation, code de conduite, best practices industrie)\n"
    "- Tax/Subsidy (taxe, subside, incitation financière)\n"
    "- Public investment & procurement (investissement public, achat/commande publique)\n"
    "- Prohibition/Ban (interdiction d'usage ou du résultat de l'usage)\n"
    "- Planning & experimentation (plan, évaluation, essai/pilote, programme)\n"
    "- Obligation (obligations/standards obligatoires techniques/éthiques; exigences de conformité)\n"
    "- Liability scheme (responsabilité civile/administrative/pénale; régime de responsabilité)\n\n"
    "SORTIE OBLIGATOIRE:\n"
    "- Réponds UNIQUEMENT avec un JSON strict, sans texte autour, sans markdown.\n"
    "- Le champ 'justification' doit venir AVANT 'targets' et 'instruments'.\n"
    "- 'targets' et 'instruments' doivent être des tableaux (liste de strings) avec uniquement les labels EXACTS.\n"
    "- Si aucun target/instrument ne s'applique: targets=[], instruments=[] (pas 'NONE').\n"
)

USER_TEMPLATE = """Analyse l'article ci-dessous.

Consigne:
1) Écris une justification courte (1-3 phrases) expliquant pourquoi les targets/instruments sélectionnés s'appliquent (ou pourquoi aucun ne s'applique).
2) Ensuite sélectionne les targets et instruments dans les listes fermées (0..n).
3) Reste très conservateur (faux négatifs > faux positifs).

Retourne UNIQUEMENT ce JSON strict (dans cet ordre de clés):
{{
  "justification": "...",
  "targets": ["..."],
  "instruments": ["..."]
}}

Texte légal:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row.get(text_col)) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)