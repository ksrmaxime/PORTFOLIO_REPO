# src/run_target_prompts.py
# Un prompt système dédié par cible (target), sur le même modèle que
# run_inst_prompts.py : une question binaire fermée par cible, au lieu d'une
# seule question générique couvrant les 12 cibles à la fois.
#
# Taxonomie des 12 cibles opérationnelles : PoC PDF, section 2.3 (nouvelle
# grille à 4 quadrants Enabling/Safeguarding x Upstream/Downstream). Cette
# taxonomie remplace celle de run5_prompts.TARGET_CODES (10 cibles, ancienne
# version) — ne pas réutiliser run5_prompts pour ce pipeline.
#
# Cadrage des prompts : un LLM à qui l'on demande directement "est-ce que
# cette norme fait X ?" a tendance à répondre OUI par similarité de
# vocabulaire avec le domaine de la cible (p. ex. "sécurité routière" ->
# SECURITY_ROBUSTNESS) sans jamais vérifier que la norme parle d'un objet
# numérique/algorithmique. Le prompt est donc structuré en deux tests
# séquentiels, dans cet ordre :
#   1. PREMIER TEST (partagé, identique pour les 12 cibles) : la norme
#      régule-t-elle, elle-même, un objet pertinent pour l'IA (traitement
#      automatisé, calcul/infrastructure numérique, système ou processus
#      algorithmique, compétences numériques) ? Si non -> NON immédiat, sans
#      poursuivre.
#   2. SECOND TEST (spécifique à la cible) : seulement si le premier test
#      est réussi, la définition précise de la cible s'applique-t-elle ?
# Cette structure force le LLM à écarter d'abord les normes qui n'ont rien
# de numérique/algorithmique, avant même de considérer la cible spécifique.
from __future__ import annotations

from collections import OrderedDict
from textwrap import dedent

import pandas as pd


def _p(text: str) -> str:
    """Aplatit un bloc triple-quoté multi-lignes en un seul paragraphe.

    Permet d'écrire un texte long comme UNE chaîne continue et lisible dans
    le source, sans le découper en fragments "..." "..." concaténés ligne
    par ligne.
    """
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Test partagé par les 12 cibles : avant même de considérer la cible
# spécifique, la norme doit réguler elle-même au moins un objet pertinent
# pour l'IA. Ce test est identique pour toutes les cibles — c'est un filtre
# de pertinence générique, pas une définition de cible.
# ---------------------------------------------------------------------------

_AI_RELEVANT_OBJECT_TEST = _p("""
    le traitement de données par des systèmes automatisés ou
    algorithmiques ; des capacités de calcul, de stockage de données, ou
    une infrastructure numérique pertinente pour des systèmes
    informatiques ; un système, processus, décision, résultat,
    développement ou usage algorithmique ou automatisé ; des compétences
    numériques, informatiques, en automatisation, ou liées à
    l'intelligence artificielle
""")

# ---------------------------------------------------------------------------
# Les 12 cibles opérationnelles, dans l'ordre du tableau 2.3 du PDF.
# `definition` : le second test, propre à chaque cible — la définition
# précise qui s'applique uniquement si le premier test (objet pertinent
# pour l'IA) est déjà réussi.
# ---------------------------------------------------------------------------

TARGET_DEFINITIONS: "OrderedDict[str, dict]" = OrderedDict(
    [
        (
            "RESEARCH_INNOVATION",
            {
                "name": "Recherche & Innovation",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme prévoit un
                    dispositif de soutien à la recherche ou à l'innovation
                    dont l'objet est spécifiquement l'intelligence
                    artificielle : financement de projets de recherche en
                    intelligence artificielle, création de centres de
                    recherche en intelligence artificielle, collaborations
                    scientifiques sur l'intelligence artificielle, ou
                    transfert de technologie portant sur l'intelligence
                    artificielle. Un dispositif de soutien à la recherche ou
                    à l'innovation en général, qui ne vise pas
                    spécifiquement l'intelligence artificielle, ne satisfait
                    pas cette cible.
                """),
            },
        ),
        (
            "SKILLS_HUMAN_CAPITAL",
            {
                "name": "Compétences & Capital humain",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme prévoit un
                    dispositif de formation ou de développement de
                    compétences dont l'objet est spécifiquement
                    l'intelligence artificielle, les données, le calcul ou
                    les systèmes automatisés : formations à l'intelligence
                    artificielle, compétences en données ou en calcul,
                    programmes universitaires spécialisés, requalification
                    professionnelle liée à l'intelligence artificielle. Un
                    dispositif d'éducation ou de formation générale, qui ne
                    vise pas spécifiquement ces compétences, ne satisfait
                    pas cette cible.
                """),
            },
        ),
        (
            "DATA_ACCESS_RESOURCES",
            {
                "name": "Accès aux données & Ressources",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme facilite l'accès,
                    le partage ou la réutilisation de données dont l'objet
                    est spécifiquement le développement de systèmes
                    d'intelligence artificielle ou le traitement automatisé.
                    Une norme qui organise la collecte ou la gestion de
                    données publiques, administratives ou statistiques en
                    général, sans lien avec le développement de systèmes
                    d'intelligence artificielle, ne satisfait pas cette
                    cible.
                """),
            },
        ),
        (
            "COMPUTE_INFRASTRUCTURE",
            {
                "name": "Calcul & Infrastructure",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme facilite l'accès
                    à des capacités de calcul ou à une infrastructure
                    (puces, matériel informatique, cloud, supercalcul,
                    centres de données) dont l'objet est spécifiquement le
                    développement ou le fonctionnement de systèmes
                    d'intelligence artificielle. Une norme qui régule
                    l'infrastructure ou l'énergie en général, sans lien avec
                    le fonctionnement de systèmes d'intelligence
                    artificielle, ne satisfait pas cette cible.
                """),
            },
        ),
        (
            "ADOPTION_DIFFUSION",
            {
                "name": "Adoption & Diffusion",
                "quadrant": "Enabling x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme encourage
                    l'adoption ou le déploiement effectif de systèmes
                    d'intelligence artificielle par des entreprises, des
                    administrations publiques ou d'autres organisations.
                """),
            },
        ),
        (
            "EXPERIMENTATION_MARKET",
            {
                "name": "Expérimentation & Développement de marché",
                "quadrant": "Enabling x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme met en place un
                    dispositif d'expérimentation encadrée (par exemple un
                    bac à sable réglementaire) ou de facilitation de
                    l'entrée sur le marché, dont l'objet est spécifiquement
                    des systèmes d'intelligence artificielle.
                """),
            },
        ),
        (
            "DATA_PRIVACY",
            {
                "name": "Données & Vie privée",
                "quadrant": "Safeguarding x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme protège des
                    droits ou intérêts liés à des données dont le traitement
                    (collecte, réutilisation, ré-identification, inférence)
                    est spécifiquement effectué par l'intelligence
                    artificielle ou des systèmes automatisés. Une norme de
                    protection des données qui ne vise pas spécifiquement un
                    traitement automatisé ou l'intelligence artificielle ne
                    satisfait pas cette cible.
                """),
            },
        ),
        (
            "IP_CREATIVE_RIGHTS",
            {
                "name": "Propriété intellectuelle & Droits créatifs",
                "quadrant": "Safeguarding x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme protège ou
                    attribue des droits de propriété intellectuelle ou des
                    droits d'auteur sur un contenu dont le lien avec
                    l'intelligence artificielle est explicite : données
                    d'entraînement, œuvres protégées utilisées pour
                    entraîner un système d'intelligence artificielle, ou
                    contenu généré par un système d'intelligence
                    artificielle.
                """),
            },
        ),
        (
            "SECURITY_ROBUSTNESS",
            {
                "name": "Sécurité & Robustesse",
                "quadrant": "Safeguarding x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme impose une
                    exigence de sécurité, d'intégrité, de résilience ou de
                    robustesse dont l'objet est spécifiquement des systèmes,
                    modèles, données ou infrastructures d'intelligence
                    artificielle. Une exigence de cybersécurité générale,
                    sans lien avec l'intelligence artificielle, ne satisfait
                    pas cette cible.
                """),
            },
        ),
        (
            "ACCOUNTABILITY_TRANSPARENCY",
            {
                "name": "Responsabilité & Transparence",
                "quadrant": "Safeguarding x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme exige de la
                    transparence, de l'explicabilité, de la traçabilité, une
                    supervision humaine ou une possibilité de contester une
                    décision, dont l'objet est spécifiquement une décision
                    issue d'un système d'intelligence artificielle ou
                    automatisé.
                """),
            },
        ),
        (
            "HIGH_STAKES_RIGHTS",
            {
                "name": "Usages à hauts enjeux & Droits fondamentaux",
                "quadrant": "Safeguarding x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme encadre
                    spécifiquement l'utilisation d'un système d'intelligence
                    artificielle dans un contexte à hauts enjeux pour les
                    individus (mobilité, emploi, crédit, santé, éducation,
                    police, justice, prestations sociales, migration,
                    discrimination).
                """),
            },
        ),
        (
            "INFORMATION_SOCIETAL_HARMS",
            {
                "name": "Information & Préjudices sociétaux",
                "quadrant": "Safeguarding x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme protège contre
                    des préjudices créés ou amplifiés par la génération, la
                    recommandation, le ciblage ou la diffusion automatisée
                    d'information par un système d'intelligence artificielle
                    (désinformation, deepfakes, manipulation automatisée).
                    Une régulation générale des médias ou de la
                    désinformation, sans lien structurel avec
                    l'automatisation ou l'intelligence artificielle, ne
                    satisfait pas cette cible.
                """),
            },
        ),
    ]
)

TARGET_CODES: "OrderedDict[str, str]" = OrderedDict(
    (code, d["name"]) for code, d in TARGET_DEFINITIONS.items()
)


def build_system_prompt(code: str) -> str:
    if code not in TARGET_DEFINITIONS:
        raise KeyError(f"Unknown target code: {code}")

    d = TARGET_DEFINITIONS[code]
    name = d["name"]

    return dedent(f"""\
        Tu es un expert en analyse des politiques publiques et du droit suisse.

        ## Tâche

        Détermine si la norme ci-dessous régule la cible « {name} », qui relève de la régulation de l'intelligence artificielle.

        NON est la réponse par défaut.

        ## Premier test — objet pertinent pour l'IA

        Avant même d'examiner la cible « {name} », détermine si la norme régule elle-même au moins un des éléments suivants : {_AI_RELEVANT_OBJECT_TEST}.

        Si la norme elle-même ne régule aucun de ces éléments, réponds NON immédiatement. Ne poursuis pas vers le second test.

        ## Second test — {name}

        Seulement si le premier test est réussi :

        {d['definition']}

        Si la norme satisfait cette définition, réponds OUI. Sinon, réponds NON.

        Réponds TOUJOURS en deux parties, dans cet ordre exact, sans aucun autre texte avant, après ou entre les deux :
        Justification: [une phrase maximum]
        Décision: OUI ou NON

        La ligne "Décision:" est OBLIGATOIRE et doit toujours être présente.
        """)


USER_TEMPLATE = """Texte :
{article_text}

Réponds à la question posée dans tes instructions.

Réponds en deux parties dans cet ordre exact :
Justification: [une phrase maximum]
Décision: OUI ou NON"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
