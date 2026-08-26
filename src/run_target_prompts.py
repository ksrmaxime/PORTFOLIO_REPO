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
# Cadrage des prompts : chaque cible a son propre récit, pas de phrase
# générique remplie par un mot-clé. Une seule tournure ("nous savons que
# l'IA soulève un enjeu majeur : X") appliquée aux 12 cibles ne veut rien
# dire une fois X substitué par des cibles aussi différentes qu'un manque de
# recherche ou un risque de désinformation. Les 4 cibles "Enabling" (l'État
# agit pour permettre/encourager le développement de l'IA) et les 8 cibles
# "Safeguarding" (l'IA fait peser un risque concret que l'État doit
# encadrer) appellent des tournures différentes :
#   - Enabling  : "Pour <objectif de politique publique>, un État doit
#                 s'assurer qu'il existe <dispositif concret>. Est-ce que
#                 la norme ci-dessous met en place un tel dispositif ?"
#   - Safeguarding : "L'intelligence artificielle fait peser un risque de
#                 <danger concret>. Est-ce que la norme ci-dessous protège
#                 contre ce risque ?"
# `context` (la mise en situation) et `question` (la question fermée) sont
# donc écrits intégralement pour chaque cible. Seule la partie sur la forme
# de la réponse (Justification/Décision) reste commune aux 12 prompts.
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
# Les 12 cibles opérationnelles, dans l'ordre du tableau 2.3 du PDF.
# `context` : la mise en situation spécifique à la cible (objectif de
# politique publique pour les cibles Enabling, risque concret pour les
# cibles Safeguarding). `question` : la question fermée correspondante.
# ---------------------------------------------------------------------------

TARGET_DEFINITIONS: "OrderedDict[str, dict]" = OrderedDict(
    [
        (
            "RESEARCH_INNOVATION",
            {
                "name": "Recherche & Innovation",
                "quadrant": "Enabling x Upstream",
                "context": _p("""
                    Pour encourager l'innovation et le développement de
                    l'intelligence artificielle, un État doit s'assurer
                    qu'il existe des dispositifs de soutien à la recherche
                    en intelligence artificielle : financement de projets
                    de recherche, création de centres de recherche,
                    collaborations scientifiques ou transfert de
                    technologie portant sur l'intelligence artificielle.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous met en place un tel
                    dispositif de soutien à la recherche ou à l'innovation
                    en intelligence artificielle ? Un dispositif de soutien
                    à la recherche ou à l'innovation en général, qui ne
                    vise pas spécifiquement l'intelligence artificielle, ne
                    répond pas à cet objectif.
                """),
            },
        ),
        (
            "SKILLS_HUMAN_CAPITAL",
            {
                "name": "Compétences & Capital humain",
                "quadrant": "Enabling x Upstream",
                "context": _p("""
                    Pour permettre le développement de l'intelligence
                    artificielle, un État doit s'assurer que la population
                    et la main-d'œuvre disposent des compétences
                    nécessaires : formations à l'intelligence artificielle,
                    compétences en données ou en calcul, programmes
                    universitaires spécialisés, requalification
                    professionnelle.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous met en place un tel
                    dispositif de formation ou de développement de
                    compétences en intelligence artificielle ? Un
                    dispositif d'éducation ou de formation générale, qui ne
                    vise pas spécifiquement des compétences en intelligence
                    artificielle, en données ou en calcul, ne répond pas à
                    cet objectif.
                """),
            },
        ),
        (
            "DATA_ACCESS_RESOURCES",
            {
                "name": "Accès aux données & Ressources",
                "quadrant": "Enabling x Upstream",
                "context": _p("""
                    Le développement de systèmes d'intelligence
                    artificielle dépend de la disponibilité de données pour
                    leur entraînement et leur fonctionnement. Pour favoriser
                    ce développement, un État peut faciliter l'accès, le
                    partage ou la réutilisation de données destinées à
                    l'intelligence artificielle.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous facilite un tel accès ou
                    partage de données au service du développement de
                    l'intelligence artificielle ? Une norme qui organise la
                    collecte ou la gestion de données publiques,
                    administratives ou statistiques en général, sans lien
                    avec le développement de systèmes d'intelligence
                    artificielle, ne répond pas à cet objectif.
                """),
            },
        ),
        (
            "COMPUTE_INFRASTRUCTURE",
            {
                "name": "Calcul & Infrastructure",
                "quadrant": "Enabling x Upstream",
                "context": _p("""
                    Le développement de systèmes d'intelligence
                    artificielle nécessite un accès à des capacités de
                    calcul et à une infrastructure adaptée : puces,
                    matériel informatique, cloud, supercalcul, centres de
                    données. Pour favoriser ce développement, un État peut
                    faciliter cet accès.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous facilite l'accès à des
                    capacités de calcul ou à une infrastructure destinée au
                    développement de l'intelligence artificielle ? Une
                    norme qui régule l'infrastructure ou l'énergie en
                    général, sans lien avec le fonctionnement de systèmes
                    d'intelligence artificielle, ne répond pas à cet
                    objectif.
                """),
            },
        ),
        (
            "ADOPTION_DIFFUSION",
            {
                "name": "Adoption & Diffusion",
                "quadrant": "Enabling x Downstream",
                "context": _p("""
                    Pour que l'intelligence artificielle produise des
                    bénéfices concrets, un État doit encourager son adoption
                    effective par les entreprises, les administrations
                    publiques ou d'autres organisations, par exemple par des
                    incitations, un accompagnement ou des programmes de
                    déploiement.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous encourage l'adoption ou
                    le déploiement effectif de l'intelligence artificielle
                    par des entreprises, des administrations publiques ou
                    d'autres organisations ?
                """),
            },
        ),
        (
            "EXPERIMENTATION_MARKET",
            {
                "name": "Expérimentation & Développement de marché",
                "quadrant": "Enabling x Downstream",
                "context": _p("""
                    Pour permettre à des systèmes d'intelligence
                    artificielle d'arriver à maturité et d'entrer sur le
                    marché, un État peut mettre en place des dispositifs
                    d'expérimentation encadrée, comme un bac à sable
                    réglementaire, permettant de tester ou de démontrer ces
                    systèmes avant leur déploiement complet.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous met en place un tel
                    dispositif d'expérimentation ou de facilitation de
                    l'entrée sur le marché de systèmes d'intelligence
                    artificielle ?
                """),
            },
        ),
        (
            "DATA_PRIVACY",
            {
                "name": "Données & Vie privée",
                "quadrant": "Safeguarding x Upstream",
                "context": _p("""
                    L'entraînement et le fonctionnement des systèmes
                    d'intelligence artificielle reposent souvent sur de
                    grandes quantités de données, ce qui expose les
                    personnes concernées à un risque d'atteinte à leur vie
                    privée : collecte excessive, réutilisation non
                    consentie, ré-identification, ou inférences intrusives
                    à partir de leurs données.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous protège les personnes
                    contre ce risque, s'agissant spécifiquement de données
                    utilisées, inférées ou traitées par l'intelligence
                    artificielle ou des systèmes automatisés ? Une norme de
                    protection des données qui ne vise pas spécifiquement un
                    traitement automatisé ou l'intelligence artificielle ne
                    répond pas à ce risque.
                """),
            },
        ),
        (
            "IP_CREATIVE_RIGHTS",
            {
                "name": "Propriété intellectuelle & Droits créatifs",
                "quadrant": "Safeguarding x Upstream",
                "context": _p("""
                    L'intelligence artificielle, notamment lorsqu'elle est
                    entraînée sur des œuvres protégées ou qu'elle génère
                    elle-même du contenu, pose un risque pour les droits de
                    propriété intellectuelle et les droits d'auteur :
                    utilisation non autorisée d'œuvres protégées comme
                    données d'entraînement, ou incertitude sur l'attribution
                    des droits sur un contenu généré par l'intelligence
                    artificielle.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous protège ou attribue des
                    droits de propriété intellectuelle ou des droits
                    d'auteur face à ce risque spécifique lié à
                    l'intelligence artificielle ?
                """),
            },
        ),
        (
            "SECURITY_ROBUSTNESS",
            {
                "name": "Sécurité & Robustesse",
                "quadrant": "Safeguarding x Upstream",
                "context": _p("""
                    Des systèmes d'intelligence artificielle défaillants,
                    non sécurisés ou peu robustes peuvent causer des
                    dommages : failles de sécurité exploitables, erreurs non
                    détectées, dérives de fonctionnement, vulnérabilité aux
                    attaques ciblant les modèles ou les données
                    d'entraînement.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous impose une exigence de
                    sécurité, d'intégrité, de résilience ou de robustesse
                    face à ce risque, s'agissant spécifiquement de systèmes,
                    modèles, données ou infrastructures d'intelligence
                    artificielle ? Une exigence de cybersécurité générale,
                    sans lien avec l'intelligence artificielle, ne répond
                    pas à ce risque.
                """),
            },
        ),
        (
            "ACCOUNTABILITY_TRANSPARENCY",
            {
                "name": "Responsabilité & Transparence",
                "quadrant": "Safeguarding x Downstream",
                "context": _p("""
                    Une décision prise ou assistée par un système
                    d'intelligence artificielle peut rester incompréhensible
                    ou invérifiable pour la personne concernée, qui risque
                    alors de ne pas pouvoir comprendre, vérifier ou
                    contester cette décision.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous protège les personnes
                    contre ce risque, en exigeant de la transparence, de
                    l'explicabilité, de la traçabilité, une supervision
                    humaine ou une possibilité de contester une décision
                    issue d'un système d'intelligence artificielle ou
                    automatisé ?
                """),
            },
        ),
        (
            "HIGH_STAKES_RIGHTS",
            {
                "name": "Usages à hauts enjeux & Droits fondamentaux",
                "quadrant": "Safeguarding x Downstream",
                "context": _p("""
                    Un système d'intelligence artificielle utilisé dans un
                    domaine à hauts enjeux (mobilité, emploi, crédit, santé,
                    éducation, police, justice, prestations sociales,
                    migration) peut produire des décisions ayant de lourdes
                    conséquences pour les individus, jusqu'à porter atteinte
                    à leurs droits fondamentaux ou à leur intégrité, par
                    exemple par des biais ou des discriminations.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous encadre spécifiquement
                    l'utilisation de l'intelligence artificielle dans un tel
                    contexte à hauts enjeux, pour protéger les individus
                    face à ce risque ?
                """),
            },
        ),
        (
            "INFORMATION_SOCIETAL_HARMS",
            {
                "name": "Information & Préjudices sociétaux",
                "quadrant": "Safeguarding x Downstream",
                "context": _p("""
                    La génération, la recommandation, le ciblage ou la
                    diffusion automatisée d'information par des systèmes
                    d'intelligence artificielle peut amplifier des
                    préjudices sociétaux : désinformation, deepfakes,
                    manipulation automatisée des opinions ou des
                    comportements.
                """),
                "question": _p("""
                    Est-ce que la norme ci-dessous protège la société contre
                    ce risque, s'agissant spécifiquement d'un contenu
                    généré, recommandé, ciblé ou diffusé de manière
                    automatisée ? Une régulation générale des médias ou de
                    la désinformation, sans lien structurel avec
                    l'automatisation ou l'intelligence artificielle, ne
                    répond pas à ce risque.
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

    return dedent(f"""\
        Tu es un expert en analyse des politiques publiques et du droit suisse.

        {d['context']}

        Voici ci-dessous une norme légale, sélectionnée aléatoirement parmi des textes qui ont, ou n'ont rien à voir, avec cette problématique.

        {d['question']}

        Réponds TOUJOURS en deux parties, dans cet ordre exact, sans aucun autre texte avant, après ou entre les deux :
        Justification: [une phrase maximum]
        Décision: OUI ou NON

        La ligne "Décision:" est OBLIGATOIRE et doit toujours être présente.
        
        Attention ne répond OUI qui quand tu es sur que la norme a été introduite pour répondre à l'objectif de politique publique ou au risque concret décrit ci-dessus. Soit conservateur !
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
