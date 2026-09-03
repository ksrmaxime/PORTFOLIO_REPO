# src/run_target_control_prompts.py
# Prompts du run de CONTRÔLE : reprend, cible par cible, uniquement les
# articles classés OUI au premier passage (run_target_prompts.py) et demande
# au LLM de confirmer ou d'infirmer cette classification.
#
# Contrairement au premier passage, la définition de la cible fournie ici est
# volontairement très succincte (une à deux phrases, sans contre-exemples ni
# ancrage détaillé) : l'objectif n'est pas de re-dérouler tout le
# raisonnement du prompt initial, mais d'obtenir un second regard,
# indépendant et rapide, sur une classification déjà posée. L'intitulé montré
# au LLM (prompt_label) reste, comme dans run_target_prompts.py, délibérément
# plus spécifique à l'IA que le nom officiel de la cible dans le document —
# le LLM ne voit qu'une seule cible à la fois et n'a aucune connaissance des
# 9 autres.
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
        "La norme organise ou finance la mise à disposition pratique d'une "
        "ressource de données pour la recherche, l'entraînement ou le test "
        "de systèmes d'IA (jeu de données ouvertes, plateforme de partage), "
        "sans poser de condition juridique sur son utilisation."
    ),
    "COMPUTE_INFRASTRUCTURE": (
        "La norme finance ou facilite l'accès à la puissance de calcul "
        "(GPU, cloud, supercalculateurs, centres de données) pour "
        "entraîner ou faire fonctionner des systèmes d'IA."
    ),
    "DATA_PRIVACY_IP": (
        "La norme régule un droit, une protection ou une condition "
        "juridique (protection des données personnelles, consentement, "
        "licence, droit d'auteur ou exception à ce droit) attaché à des "
        "données ou des contenus utilisés en amont pour développer, "
        "entraîner ou faire fonctionner un système d'IA."
    ),
    "SECURITY_ROBUSTNESS": (
        "La norme impose une exigence visant à protéger l'intégrité "
        "technique (sécurité, fiabilité, résilience, robustesse) d'un "
        "système d'IA contre une attaque malveillante ou une défaillance."
    ),
    "AI_DEPLOYMENT": (
        "La norme encourage, finance, autorise ou facilite concrètement "
        "l'utilisation ou le déploiement d'un système d'IA déjà existant "
        "par une entreprise, une administration ou une organisation (y "
        "compris à titre pilote ou via un bac à sable réglementaire)."
    ),
    "ACCOUNTABILITY_TRANSPARENCY": (
        "La norme rend le fonctionnement, l'usage ou la décision d'un "
        "système d'IA transparent, traçable, explicable, contrôlable, "
        "contestable, ou attribuable à un acteur responsable (information "
        "de la personne concernée, explication, supervision humaine, "
        "traçabilité, droit de recours, ou publication de caractéristiques "
        "techniques)."
    ),
    "OUTPUT_HARMS": (
        "La norme prévient, restreint, corrige ou offre un remède contre "
        "un résultat concret (contenu, décision, prédiction, "
        "recommandation, action) directement produit par un système d'IA, "
        "lorsque ce résultat est dommageable, illicite, dangereux, "
        "discriminatoire ou trompeur."
    ),
    "SOCIETAL_HARMS": (
        "La norme s'attaque à une conséquence collective, systémique ou "
        "sociétale de l'usage généralisé de l'IA (désinformation à grande "
        "échelle, manipulation électorale, discrimination systémique, "
        "risques institutionnels ou de marché) plutôt qu'à un résultat "
        "individuel isolé."
    ),
}


def build_system_prompt(code: str) -> str:
    if code not in TARGET_DEFINITIONS:
        raise KeyError(f"Unknown target code: {code}")
    if code not in SHORT_TARGET_DEFINITIONS:
        raise KeyError(f"No short definition for target code: {code}")

    label = TARGET_DEFINITIONS[code]["prompt_label"]
    short_def = SHORT_TARGET_DEFINITIONS[code]

    blocks = [
        "Tu es un expert en analyse des politiques publiques et du droit "
        "suisse. Ta tâche est de vérifier, en second avis, une "
        "classification déjà posée par un autre modèle de langage.",
        f"Problème public évalué : « {label} ».\n\n{short_def}",
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
    name = TARGET_DEFINITIONS[code]["prompt_label"]
    return USER_TEMPLATE.format(article_text=txt, target_name=name)
