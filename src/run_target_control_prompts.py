# src/run_target_control_prompts.py
# Prompts du run de CONTRÔLE : reprend, cible par cible, uniquement les
# articles classés OUI au premier passage (run_target_prompts.py) et demande
# au LLM de confirmer ou d'infirmer cette classification.
#
# Contrairement au premier passage, la définition de la cible fournie ici est
# volontairement très succincte (une à deux phrases, sans contre-exemples ni
# ancrage détaillé) : l'objectif n'est pas de re-dérouler tout le
# raisonnement du prompt initial, mais d'obtenir un second regard,
# indépendant et rapide, sur une classification déjà posée.
from __future__ import annotations

import pandas as pd

from src.run_target_prompts import TARGET_DEFINITIONS

# Une définition très courte par cible — dérivée de TARGET_DEFINITIONS mais
# réduite à l'essentiel. Les clés doivent rester synchronisées avec
# TARGET_DEFINITIONS dans src/run_target_prompts.py.
SHORT_TARGET_DEFINITIONS: dict[str, str] = {
    "RESEARCH_INNOVATION": (
        "L'État finance ou organise concrètement une activité de recherche "
        "en intelligence artificielle (financement de projets, création "
        "d'un centre de recherche, subvention à la R&D)."
    ),
    "SKILLS_HUMAN_CAPITAL": (
        "L'État organise ou finance un enseignement ou une formation dont "
        "le contenu porte spécifiquement sur l'intelligence artificielle, "
        "la science des données ou le calcul informatique."
    ),
    "DATA_ACCESS_RESOURCES": (
        "La norme facilite l'accès à des données pour entraîner des "
        "systèmes d'IA — mise à disposition de données, ou suppression "
        "d'une protection (ex. droit d'auteur) qui les rend réutilisables "
        "pour l'entraînement."
    ),
    "COMPUTE_INFRASTRUCTURE": (
        "La norme finance ou facilite l'accès à la puissance de calcul "
        "(GPU, cloud, supercalculateurs, centres de données) pour "
        "entraîner ou faire fonctionner des systèmes d'IA."
    ),
    "ADOPTION_DIFFUSION": (
        "La norme encourage, finance, autorise ou facilite concrètement "
        "l'utilisation ou le déploiement d'un système d'IA déjà existant "
        "par une entreprise, une administration ou une organisation (y "
        "compris à titre pilote ou via un bac à sable réglementaire)."
    ),
    "DATA_PRIVACY": (
        "La norme régule le traitement de données personnelles (collecte, "
        "conservation, réutilisation) dans le cadre de l'utilisation d'un "
        "système d'IA ou d'un traitement automatisé, de façon générale ou "
        "spécifique à l'IA."
    ),
    "IP_CREATIVE_RIGHTS": (
        "La norme régule des droits de propriété intellectuelle ou "
        "d'auteur portant sur un contenu généré EN SORTIE par un système "
        "d'IA (titularité, protection, contrefaçon)."
    ),
    "SECURITY_ROBUSTNESS": (
        "La norme impose une exigence visant à protéger un système d'IA "
        "(ou automatisé) contre une cyberattaque : intrusion, piratage, "
        "manipulation malveillante, sabotage."
    ),
    "ACCOUNTABILITY_TRANSPARENCY": (
        "La norme impose (a) d'informer une personne qu'une décision la "
        "concernant a été prise par un système d'IA, ou (b) de communiquer "
        "des caractéristiques techniques d'un système d'IA."
    ),
    "HIGH_STAKES_RISKS": (
        "La norme régule directement l'utilisation d'un système d'IA dans "
        "un contexte à hauts enjeux (droits fondamentaux, sécurité "
        "physique) ou informationnel (désinformation, manipulation), dans "
        "le but d'en réduire le risque."
    ),
}


def build_system_prompt(code: str) -> str:
    if code not in TARGET_DEFINITIONS:
        raise KeyError(f"Unknown target code: {code}")
    if code not in SHORT_TARGET_DEFINITIONS:
        raise KeyError(f"No short definition for target code: {code}")

    name = TARGET_DEFINITIONS[code]["name"]
    short_def = SHORT_TARGET_DEFINITIONS[code]

    blocks = [
        "Tu es un expert en analyse des politiques publiques et du droit "
        "suisse. Ta tâche est de vérifier, en second avis, une "
        "classification déjà posée par un autre modèle de langage.",
        f"Problème public évalué : « {name} ».\n\n{short_def}",
        "Réponds TOUJOURS en deux parties, dans cet ordre exact, sans "
        "aucun autre texte avant, après ou entre les deux :\n"
        "Justification: [une phrase maximum]\n"
        "Décision: NON ou OUI\n\n"
        'La ligne "Décision:" est OBLIGATOIRE et doit toujours être '
        "présente.",
    ]
    return "\n\n".join(blocks)


USER_TEMPLATE = """Texte :
{article_text}

La mesure ci-dessus a été classifiée par un autre modèle comme répondant au problème public « {target_name} ». Peux-tu confirmer cette classification par oui ou non ?

Réponds en deux parties dans cet ordre exact :
Justification: [une phrase maximum]
Décision: NON ou OUI"""


def build_user_prompt(row: pd.Series, text_col: str, code: str) -> str:
    if code not in TARGET_DEFINITIONS:
        raise KeyError(f"Unknown target code: {code}")
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    name = TARGET_DEFINITIONS[code]["name"]
    return USER_TEMPLATE.format(article_text=txt, target_name=name)
