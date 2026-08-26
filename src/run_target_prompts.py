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
# Cadrage des prompts : mise en situation, pas de liste de règles. Les
# versions précédentes empilaient un gros bloc de règles générales
# (anti-extrapolation, mots interdits, exemples, gestion des alinéas...) qui
# prenait le dessus sur la question réellement posée pour la cible — le LLM
# finissait par raisonner sur les règles plutôt que sur le texte. Le prompt
# est maintenant réduit à : une mise en situation courte rappelant l'enjeu
# IA visé par la cible, la norme à évaluer, et la question fermée. On laisse
# la compréhension contextuelle au LLM plutôt que de la contraindre par une
# liste de règles — c'est tout l'intérêt de passer par un LLM plutôt qu'une
# recherche de mots-clés.
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

# ---------------------------------------------------------------------------
# Les 12 cibles opérationnelles, dans l'ordre du tableau 2.3 du PDF.
# Chaque entrée : nom, quadrant, et `probleme` — une description courte de
# l'enjeu que l'intelligence artificielle pose pour cette cible, utilisée
# dans la mise en situation du prompt.
# ---------------------------------------------------------------------------

TARGET_DEFINITIONS: "OrderedDict[str, dict]" = OrderedDict(
    [
        (
            "RESEARCH_INNOVATION",
            {
                "name": "Recherche & Innovation",
                "quadrant": "Enabling x Upstream",
                "probleme": (
                    "un manque de recherche et d'innovation en intelligence "
                    "artificielle, faute de financement, de centres de recherche "
                    "ou de collaboration scientifique dédiés"
                ),
            },
        ),
        (
            "SKILLS_HUMAN_CAPITAL",
            {
                "name": "Compétences & Capital humain",
                "quadrant": "Enabling x Upstream",
                "probleme": (
                    "un manque de compétences humaines pertinentes pour "
                    "l'intelligence artificielle (formation, compétences en "
                    "données ou en calcul, main-d'œuvre qualifiée)"
                ),
            },
        ),
        (
            "DATA_ACCESS_RESOURCES",
            {
                "name": "Accès aux données & Ressources",
                "quadrant": "Enabling x Upstream",
                "probleme": (
                    "un accès insuffisant aux données nécessaires au "
                    "développement de systèmes d'intelligence artificielle"
                ),
            },
        ),
        (
            "COMPUTE_INFRASTRUCTURE",
            {
                "name": "Calcul & Infrastructure",
                "quadrant": "Enabling x Upstream",
                "probleme": (
                    "un accès insuffisant à des capacités de calcul ou à une "
                    "infrastructure adaptée au développement de l'intelligence "
                    "artificielle"
                ),
            },
        ),
        (
            "ADOPTION_DIFFUSION",
            {
                "name": "Adoption & Diffusion",
                "quadrant": "Enabling x Downstream",
                "probleme": (
                    "une adoption insuffisante de l'intelligence artificielle "
                    "par les entreprises, les administrations publiques ou "
                    "d'autres organisations"
                ),
            },
        ),
        (
            "EXPERIMENTATION_MARKET",
            {
                "name": "Expérimentation & Développement de marché",
                "quadrant": "Enabling x Downstream",
                "probleme": (
                    "des difficultés à tester, expérimenter ou mettre sur le "
                    "marché des systèmes d'intelligence artificielle"
                ),
            },
        ),
        (
            "DATA_PRIVACY",
            {
                "name": "Données & Vie privée",
                "quadrant": "Safeguarding x Upstream",
                "probleme": (
                    "la protection des données utilisées, inférées ou traitées "
                    "par des systèmes d'intelligence artificielle"
                ),
            },
        ),
        (
            "IP_CREATIVE_RIGHTS",
            {
                "name": "Propriété intellectuelle & Droits créatifs",
                "quadrant": "Safeguarding x Upstream",
                "probleme": (
                    "la protection des droits de propriété intellectuelle ou "
                    "des droits d'auteur sur du contenu affecté par "
                    "l'intelligence artificielle (données d'entraînement, "
                    "œuvres protégées, contenu généré)"
                ),
            },
        ),
        (
            "SECURITY_ROBUSTNESS",
            {
                "name": "Sécurité & Robustesse",
                "quadrant": "Safeguarding x Upstream",
                "probleme": (
                    "la sécurité, l'intégrité ou la robustesse des systèmes, "
                    "modèles ou infrastructures d'intelligence artificielle"
                ),
            },
        ),
        (
            "ACCOUNTABILITY_TRANSPARENCY",
            {
                "name": "Responsabilité & Transparence",
                "quadrant": "Safeguarding x Downstream",
                "probleme": (
                    "le manque de transparence, d'explicabilité, de "
                    "traçabilité ou de supervision humaine des décisions "
                    "prises par des systèmes d'intelligence artificielle"
                ),
            },
        ),
        (
            "HIGH_STAKES_RIGHTS",
            {
                "name": "Usages à hauts enjeux & Droits fondamentaux",
                "quadrant": "Safeguarding x Downstream",
                "probleme": (
                    "les conséquences lourdes que peuvent avoir, pour les "
                    "individus et leurs droits fondamentaux, des décisions "
                    "prises par des systèmes d'intelligence artificielle dans "
                    "des domaines à hauts enjeux (mobilité, emploi, crédit, "
                    "santé, éducation, police, justice, prestations sociales, "
                    "migration, discrimination)"
                ),
            },
        ),
        (
            "INFORMATION_SOCIETAL_HARMS",
            {
                "name": "Information & Préjudices sociétaux",
                "quadrant": "Safeguarding x Downstream",
                "probleme": (
                    "les préjudices sociétaux créés ou amplifiés par la "
                    "génération, la recommandation, le ciblage ou la diffusion "
                    "automatisée d'information (désinformation, deepfakes, "
                    "manipulation)"
                ),
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

    return (
        "Tu es un expert en analyse des politiques publiques et du droit suisse.\n\n"

        f"Nous savons que l'intelligence artificielle soulève un enjeu majeur : "
        f"{d['probleme']}.\n\n"

        "Voici ci-dessous une norme légale, sélectionnée aléatoirement parmi des "
        "textes qui ont, ou n'ont rien à voir, avec cette problématique.\n\n"

        "Dis-moi si, oui ou non, cette norme répond à ce problème.\n\n"

        "Réponds TOUJOURS en deux parties, dans cet ordre exact, sans aucun autre "
        "texte avant, après ou entre les deux :\n"
        "Justification: [une phrase maximum]\n"
        "Décision: OUI ou NON\n\n"
        "La ligne \"Décision:\" est OBLIGATOIRE et doit toujours être présente."
    )


USER_TEMPLATE = """Texte :
{article_text}

Réponds à la question posée dans tes instructions.

Réponds en deux parties dans cet ordre exact :
Justification: [une phrase maximum]
Décision: OUI ou NON"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
