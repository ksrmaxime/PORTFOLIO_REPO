# src/run_target_prompts.py
# Un prompt système dédié par cible (target), sur le même modèle que
# run_inst_prompts.py : une question binaire fermée par cible, au lieu d'une
# seule question générique couvrant les 12 cibles à la fois.
#
# Taxonomie des 12 cibles opérationnelles : PoC PDF, section 2.3 (nouvelle
# grille à 4 quadrants Enabling/Safeguarding x Upstream/Downstream). Cette
# taxonomie remplace celle de run5_prompts.TARGET_CODES (10 cibles, ancienne
# version) — ne pas réutiliser run5_prompts pour ce pipeline.
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

# ---------------------------------------------------------------------------
# Les 12 cibles opérationnelles, dans l'ordre du tableau 2.3 du PDF.
# Chaque entrée : (nom, quadrant, question fermée, note optionnelle sur les
# exclusions/pièges propres à cette cible).
# ---------------------------------------------------------------------------

def _p(text: str) -> str:
    """Normalise un bloc de texte multi-lignes en un seul paragraphe.

    Permet d'écrire question/note comme UNE chaîne continue dans le source
    (repliée sur plusieurs lignes pour la lisibilité), sans risquer les bugs
    de concaténation implicite entre fragments "" (espaces manquants/en trop,
    ponctuation collée) qui font que le LLM lit chaque ligne comme une
    consigne séparée au lieu d'une seule phrase.
    """
    return " ".join(text.split())


TARGET_DEFINITIONS: "OrderedDict[str, dict]" = OrderedDict(
    [
        (
            "RESEARCH_INNOVATION",
            {
                "name": "Recherche & Innovation",
                "quadrant": "Enabling x Upstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement des mesures augmentant la
                    recherche ou l'innovation en intelligence artificielle (comme le
                    financement de recherche, centres de recherche, collaboration
                    scientifique, transfert de technologie portant directement sur
                    l'intelligence artificielle) ? Si l'article prévoit des mesures
                    soutenant l'innovation en général, mais pas explicitement
                    l'intelligence artificielle, la réponse est NON.
                    """
                ),
            },
        ),
        (
            "SKILLS_HUMAN_CAPITAL",
            {
                "name": "Compétences & Capital humain",
                "quadrant": "Enabling x Upstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement des mesures développant
                    des compétences humaines pertinentes pour l'intelligence
                    artificielle (comme des formations à l'intelligence artificielle,
                    compétences en données ou en calcul, formation spécialisée,
                    développement de la main-d'œuvre, programmes universitaires en lien
                    avec l'intelligence artificielle) ?
                    """
                ),
                "note": _p(
                    """
                    L'éducation générale est exclue, sauf si le contenu ou la
                    compétence visée est explicitement lié à l'intelligence
                    artificielle, aux données, au calcul ou aux systèmes automatisés.
                    """
                ),
            },
        ),
        (
            "DATA_ACCESS_RESOURCES",
            {
                "name": "Accès aux données & Ressources",
                "quadrant": "Enabling x Upstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement de faciliter l'accès, le
                    partage, la disponibilité ou la réutilisation de données pour le
                    développement de l'intelligence artificielle ou le traitement
                    automatisé ?
                    """
                ),
                "note": _p(
                    """
                    La simple collecte de données publiques, administratives ou
                    statistiques générales, sans lien explicite avec l'intelligence
                    artificielle ou le traitement automatisé, ne suffit pas pour répondre OUI.
                    """
                ),
            },
        ),
        (
            "COMPUTE_INFRASTRUCTURE",
            {
                "name": "Calcul & Infrastructure",
                "quadrant": "Enabling x Upstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement une mesure facilitant
                    l'accès à des capacités de calcul ou à une infrastructure physique
                    pertinente pour l'intelligence artificielle (puces, matériel,
                    cloud, supercalcul, centres de données) ?
                    """
                ),
                "note": _p(
                    """
                    Les règles générales d'infrastructure ou d'énergie sont exclues,
                    sauf si elles conditionnent matériellement le fonctionnement de
                    systèmes d'intelligence artificielle.
                    """
                ),
            },
        ),
        (
            "ADOPTION_DIFFUSION",
            {
                "name": "Adoption & Diffusion",
                "quadrant": "Enabling x Downstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement une mesure encourageant
                    l'adoption ou le déploiement effectif de l'intelligence
                    artificielle par des entreprises, des administrations publiques ou
                    d'autres organisations ?
                    Le but de l'article doit être la facilitation de l'adoption de l'intelligence artificielle pour répondre OUI. 
                    """
                ),
            },
        ),
        (
            "EXPERIMENTATION_MARKET",
            {
                "name": "Expérimentation & Développement de marché",
                "quadrant": "Enabling x Downstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement un dispositif permettant
                    de tester, d'expérimenter, de démontrer ou de faciliter l'entrée
                    sur le marché de systèmes d'intelligence artificielle (par exemple
                    un bac à sable réglementaire) ?
                    le but de l'expérimentation doit être de faciliter l'entrée sur le marché de l'intelligence artificielle pour répondre OUI.
                    """
                ),
            },
        ),
        (
            "DATA_PRIVACY",
            {
                "name": "Données & Vie privée",
                "quadrant": "Safeguarding x Upstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement une protection des
                    droits ou intérêts liés à des données utilisées, inférées,
                    réutilisées ou traitées par l'intelligence artificielle ou des
                    systèmes automatisés ?
                    Le but de la protection des données doit être lié à l'intelligence artificielle pour répondre OUI.
                    """
                ),
                "note": _p(
                    """
                    La protection générale des données n'est incluse que si elle
                    concerne matériellement un traitement automatisé ou des pratiques
                    de données liées à l'intelligence artificielle.
                    """
                ),
            },
        ),
        (
            "IP_CREATIVE_RIGHTS",
            {
                "name": "Propriété intellectuelle & Droits créatifs",
                "quadrant": "Safeguarding x Upstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement une protection ou une
                    attribution de droits de propriété intellectuelle ou de droits
                    d'auteur concernant du contenu affecté par l'intelligence
                    artificielle (données d'entraînement, œuvres protégées, contenu
                    généré par l'intelligence artificielle) ?
                    """
                ),
            },
        ),
        (
            "SECURITY_ROBUSTNESS",
            {
                "name": "Sécurité & Robustesse",
                "quadrant": "Safeguarding x Upstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement une exigence de
                    sécurité, d'intégrité, de résilience ou de robustesse pour des
                    systèmes, modèles, données ou infrastructures d'intelligence
                    artificielle ?
                    """
                ),
                "note": _p(
                    """
                    La cybersécurité générale est exclue, sauf si elle concerne
                    matériellement l'intelligence artificielle ou des systèmes
                    automatisés.
                    """
                ),
            },
        ),
        (
            "ACCOUNTABILITY_TRANSPARENCY",
            {
                "name": "Responsabilité & Transparence",
                "quadrant": "Safeguarding x Downstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement une exigence de
                    transparence, d'explicabilité, de traçabilité, de supervision
                    humaine ou de possibilité de contester une décision issue d'un
                    système d'intelligence artificielle ou automatisé ?
                    """
                ),
            },
        ),
        (
            "HIGH_STAKES_RIGHTS",
            {
                "name": "Usages à hauts enjeux & Droits fondamentaux",
                "quadrant": "Safeguarding x Downstream",
                "question": _p(
                    """
                    Est-ce que l'article encadre explicitement l'utilisation de
                    l'intelligence artificielle dans un contexte où les décisions produites par ces systemes automatisés
                    peuvent avoir de lourdes conséquences pour les individus, leurs intérêts et intégrité physique (mobilité, emploi, crédit, santé,
                    éducation, police, justice, prestations sociales, migration,
                    discrimination) ?
                    """
                ),
            },
        ),
        (
            "INFORMATION_SOCIETAL_HARMS",
            {
                "name": "Information & Préjudices sociétaux",
                "quadrant": "Safeguarding x Downstream",
                "question": _p(
                    """
                    Est-ce que l'article prévoit explicitement une mesure contre des
                    préjudices créés ou amplifiés par la génération, la
                    recommandation, le ciblage ou la diffusion automatisée
                    d'information (désinformation, deepfakes, manipulation
                    automatisée) ?
                    """
                ),
                "note": _p(
                    """
                    La régulation générale des médias ou de la désinformation est
                    exclue en l'absence de lien structurel explicite avec
                    l'automatisation ou l'intelligence artificielle.
                    """
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
    note_block = f"\nAttention : {d['note']}\n" if d.get("note") else ""

    return (
        "Tu es un expert en analyse des politiques publiques et du droit suisse.\n\n"

        "## Question\n\n"
        f"{d['question']}\n"
        f"{note_block}\n"

        "## Règle générale\n\n"
        "- Base ta réponse UNIQUEMENT sur ce qui est écrit explicitement dans le "
        "texte. Il est interdit d'extrapoler : soit la réponse à la question est "
        "explicitement dans le texte, soit la réponse est NON.\n"
        "- Un article de loi suisse contient souvent plusieurs alinéas : il suffit "
        "qu'UN SEUL alinéa réponde explicitement OUI à la question pour que "
        "l'article entier soit classé OUI.\n"
        "- En cas de doute, réponds NON.\n\n"

        "Réponds TOUJOURS en deux parties, dans cet ordre exact, sans aucun autre "
        "texte avant, après ou entre les deux :\n"
        "Justification: [une phrase maximum, citant ou paraphrasant le passage du "
        "texte]\n"
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
