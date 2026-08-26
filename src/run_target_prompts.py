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
# Cadrage des questions : question de BUT, pas de thème. La première version
# demandait "est-ce que l'article prévoit une mesure X" — une question
# thématique à laquelle le LLM pouvait rattacher presque n'importe quel
# article par un lien alambiqué (toute mention de données -> DATA_ACCESS,
# toute mention de sécurité -> SECURITY_ROBUSTNESS, etc.), d'où un taux de
# OUI proche de 100%. La question posée porte maintenant sur l'INTENTION de
# l'article : a-t-il été écrit dans le but de réguler l'intelligence
# artificielle pour atteindre l'objectif de la cible ? Un article qui
# s'applique incidemment à l'IA sans avoir été conçu pour elle répond NON.
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

# ---------------------------------------------------------------------------
# Les 12 cibles opérationnelles, dans l'ordre du tableau 2.3 du PDF.
# Chaque entrée : (nom, quadrant, objectif de la cible, question fermée
# posée en termes de BUT, note optionnelle sur les exclusions/pièges propres
# à cette cible).
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


def _purpose_question(objectif: str) -> str:
    """Construit la question fermée de BUT commune à toutes les cibles.

    `objectif` décrit ce que l'article doit viser (le "pour quoi") pour
    répondre OUI. La question isole explicitement le but de l'article de son
    thème ou de ses effets possibles, pour éviter que le LLM ne réponde OUI
    sur la seule base d'une mention incidente de l'IA.
    """
    return _p(
        f"""
        Est-ce que cet article a été écrit dans le BUT de réguler
        l'intelligence artificielle pour {objectif} ? Il ne s'agit pas de
        savoir si l'article pourrait un jour s'appliquer à un système
        d'intelligence artificielle, ni s'il mentionne un thème connexe
        (données, calcul, algorithme, automatisation) en passant. Il s'agit
        de savoir si l'objet même de la disposition — la raison pour
        laquelle elle a été écrite — est de réguler l'intelligence
        artificielle pour cette fin précise.
        """
    )


TARGET_DEFINITIONS: "OrderedDict[str, dict]" = OrderedDict(
    [
        (
            "RESEARCH_INNOVATION",
            {
                "name": "Recherche & Innovation",
                "quadrant": "Enabling x Upstream",
                "question": _purpose_question(
                    "augmenter la recherche ou l'innovation en intelligence "
                    "artificielle (financement de recherche, centres de "
                    "recherche, collaboration scientifique, transfert de "
                    "technologie portant directement sur l'intelligence "
                    "artificielle)"
                ),
                "note": _p(
                    """
                    Si le but de l'article est de soutenir l'innovation ou la
                    recherche en général, sans que l'intelligence artificielle
                    soit l'objet visé, la réponse est NON — même si l'IA
                    pourrait en bénéficier accessoirement.
                    """
                ),
            },
        ),
        (
            "SKILLS_HUMAN_CAPITAL",
            {
                "name": "Compétences & Capital humain",
                "quadrant": "Enabling x Upstream",
                "question": _purpose_question(
                    "développer des compétences humaines pertinentes pour "
                    "l'intelligence artificielle (formations à l'intelligence "
                    "artificielle, compétences en données ou en calcul, "
                    "formation spécialisée, développement de la main-d'œuvre, "
                    "programmes universitaires en lien avec l'intelligence "
                    "artificielle)"
                ),
                "note": _p(
                    """
                    Si le but de l'article est l'éducation ou la formation en
                    général, sans que l'intelligence artificielle, les données,
                    le calcul ou les systèmes automatisés soient l'objet
                    explicitement visé, la réponse est NON.
                    """
                ),
            },
        ),
        (
            "DATA_ACCESS_RESOURCES",
            {
                "name": "Accès aux données & Ressources",
                "quadrant": "Enabling x Upstream",
                "question": _purpose_question(
                    "faciliter l'accès, le partage, la disponibilité ou la "
                    "réutilisation de données pour le développement de "
                    "l'intelligence artificielle ou le traitement automatisé"
                ),
                "note": _p(
                    """
                    Si le but de l'article est la collecte ou la gestion de
                    données publiques, administratives ou statistiques en
                    général, sans que le développement de l'intelligence
                    artificielle ou le traitement automatisé soit l'objectif
                    visé, la réponse est NON.
                    """
                ),
            },
        ),
        (
            "COMPUTE_INFRASTRUCTURE",
            {
                "name": "Calcul & Infrastructure",
                "quadrant": "Enabling x Upstream",
                "question": _purpose_question(
                    "faciliter l'accès à des capacités de calcul ou à une "
                    "infrastructure physique pertinente pour l'intelligence "
                    "artificielle (puces, matériel, cloud, supercalcul, "
                    "centres de données)"
                ),
                "note": _p(
                    """
                    Si le but de l'article est de régler l'infrastructure ou
                    l'énergie en général, sans que le fonctionnement de
                    systèmes d'intelligence artificielle soit l'objectif
                    visé, la réponse est NON.
                    """
                ),
            },
        ),
        (
            "ADOPTION_DIFFUSION",
            {
                "name": "Adoption & Diffusion",
                "quadrant": "Enabling x Downstream",
                "question": _purpose_question(
                    "encourager l'adoption ou le déploiement effectif de "
                    "l'intelligence artificielle par des entreprises, des "
                    "administrations publiques ou d'autres organisations"
                ),
            },
        ),
        (
            "EXPERIMENTATION_MARKET",
            {
                "name": "Expérimentation & Développement de marché",
                "quadrant": "Enabling x Downstream",
                "question": _purpose_question(
                    "permettre de tester, d'expérimenter, de démontrer ou de "
                    "faciliter l'entrée sur le marché de systèmes "
                    "d'intelligence artificielle (par exemple un bac à sable "
                    "réglementaire)"
                ),
            },
        ),
        (
            "DATA_PRIVACY",
            {
                "name": "Données & Vie privée",
                "quadrant": "Safeguarding x Upstream",
                "question": _purpose_question(
                    "protéger des droits ou intérêts liés à des données "
                    "utilisées, inférées, réutilisées ou traitées par "
                    "l'intelligence artificielle ou des systèmes automatisés"
                ),
                "note": _p(
                    """
                    Si le but de l'article est la protection des données en
                    général, sans que le traitement automatisé ou des
                    pratiques de données liées à l'intelligence artificielle
                    soient l'objectif visé, la réponse est NON.
                    """
                ),
            },
        ),
        (
            "IP_CREATIVE_RIGHTS",
            {
                "name": "Propriété intellectuelle & Droits créatifs",
                "quadrant": "Safeguarding x Upstream",
                "question": _purpose_question(
                    "protéger ou attribuer des droits de propriété "
                    "intellectuelle ou des droits d'auteur concernant du "
                    "contenu affecté par l'intelligence artificielle (données "
                    "d'entraînement, œuvres protégées, contenu généré par "
                    "l'intelligence artificielle)"
                ),
            },
        ),
        (
            "SECURITY_ROBUSTNESS",
            {
                "name": "Sécurité & Robustesse",
                "quadrant": "Safeguarding x Upstream",
                "question": _purpose_question(
                    "garantir la sécurité, l'intégrité, la résilience ou la "
                    "robustesse de systèmes, modèles, données ou "
                    "infrastructures d'intelligence artificielle"
                ),
                "note": _p(
                    """
                    Si le but de l'article est la cybersécurité en général,
                    sans que l'intelligence artificielle ou des systèmes
                    automatisés soient l'objectif visé, la réponse est NON.
                    """
                ),
            },
        ),
        (
            "ACCOUNTABILITY_TRANSPARENCY",
            {
                "name": "Responsabilité & Transparence",
                "quadrant": "Safeguarding x Downstream",
                "question": _purpose_question(
                    "garantir la transparence, l'explicabilité, la "
                    "traçabilité, la supervision humaine ou la possibilité de "
                    "contester une décision issue d'un système d'intelligence "
                    "artificielle ou automatisé"
                ),
            },
        ),
        (
            "HIGH_STAKES_RIGHTS",
            {
                "name": "Usages à hauts enjeux & Droits fondamentaux",
                "quadrant": "Safeguarding x Downstream",
                "question": _purpose_question(
                    "encadrer l'utilisation de l'intelligence artificielle "
                    "dans un contexte où les décisions produites par ces "
                    "systèmes automatisés peuvent avoir de lourdes "
                    "conséquences pour les individus, leurs intérêts et leur "
                    "intégrité physique (mobilité, emploi, crédit, santé, "
                    "éducation, police, justice, prestations sociales, "
                    "migration, discrimination)"
                ),
            },
        ),
        (
            "INFORMATION_SOCIETAL_HARMS",
            {
                "name": "Information & Préjudices sociétaux",
                "quadrant": "Safeguarding x Downstream",
                "question": _purpose_question(
                    "prévenir des préjudices créés ou amplifiés par la "
                    "génération, la recommandation, le ciblage ou la "
                    "diffusion automatisée d'information (désinformation, "
                    "deepfakes, manipulation automatisée)"
                ),
                "note": _p(
                    """
                    Si le but de l'article est la régulation des médias ou de
                    la désinformation en général, sans que l'automatisation ou
                    l'intelligence artificielle soient l'objectif visé, la
                    réponse est NON.
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

        "## Ce qui compte : le BUT de l'article, pas son thème\n\n"
        "- La question ne porte PAS sur le thème de l'article ni sur ses effets "
        "possibles, mais sur son BUT explicite : cet article a-t-il été écrit POUR "
        "réguler l'intelligence artificielle et atteindre l'objectif ci-dessus ?\n"
        "- Un article qui mentionne l'intelligence artificielle, les données, le "
        "calcul, les algorithmes ou l'automatisation seulement en passant — sans que "
        "ce soit l'objet de la disposition — ne répond PAS OUI à la question. Le "
        "fait qu'un article puisse, en pratique, s'appliquer à un système d'IA ne "
        "signifie pas qu'il a été écrit dans ce but.\n"
        "- Teste-toi ainsi : si l'intelligence artificielle n'existait pas, cet "
        "article aurait-il été écrit de la même façon, pour la même raison ? Si "
        "oui, la réponse est NON — l'article ne vise pas l'intelligence "
        "artificielle comme objectif, il la couvre seulement incidemment.\n\n"

        "## Règle générale\n\n"
        "- Base ta réponse UNIQUEMENT sur ce qui est écrit explicitement dans le "
        "texte. Il est interdit d'extrapoler : soit le BUT de l'article correspond "
        "explicitement à la question posée, soit la réponse est NON.\n"
        "- Un OUI n'est valide QUE si le texte contient, en toutes lettres, un "
        "terme explicite lié au sujet de la question (p. ex. « intelligence "
        "artificielle », « IA », « système automatisé », « algorithme », selon le "
        "cas) ET que ce terme désigne l'objet même de la disposition — pas un "
        "exemple parmi d'autres, ni un cas d'application possible parmi tant "
        "d'autres. L'absence de ce terme, ou sa présence purement incidente, "
        "signifie NON.\n"
        "- Il est STRICTEMENT INTERDIT de justifier un OUI par un raisonnement "
        "conditionnel ou hypothétique du type « cela peut/pourrait inclure X si "
        "Y », « dans la mesure où cela concerne aussi... », « ce qui pourrait "
        "s'appliquer à... ». Si ta justification contient « peut », « pourrait », "
        "« si cela concerne » ou une formulation équivalente pour établir le lien "
        "avec l'intelligence artificielle, c'est que tu extrapoles : corrige ta "
        "réponse en NON.\n"
        "- Un article de loi suisse contient souvent plusieurs alinéas : il suffit "
        "qu'UN SEUL alinéa ait explicitement pour BUT de répondre OUI à la question "
        "pour que l'article entier soit classé OUI. Mais chaque alinéa doit être "
        "évalué selon les mêmes règles strictes ci-dessus — un alinéa dont "
        "l'intelligence artificielle n'est pas l'objet explicite ne compte pas.\n"
        "- En cas de doute, réponds NON.\n\n"

        "## Exemple d'erreur à éviter\n\n"
        "Texte (extrait, loi sur la circulation routière) : « Le Conseil fédéral "
        "peut admettre un dépassement de la longueur maximale et du poids maximal "
        "autorisés pour les véhicules et ensembles de véhicules qui présentent des "
        "caractéristiques de construction et d'équipement spéciales à des fins "
        "écologiques. »\n"
        "Réponse INTERDITE (extrapolation) : Décision OUI, avec la justification "
        "« ce qui peut inclure des mesures soutenant l'innovation en intelligence "
        "artificielle si ces caractéristiques sont liées à l'IA ». Cette réponse "
        "est fausse : le but de cet article est la circulation routière et la "
        "protection de l'environnement, pas l'intelligence artificielle — le texte "
        "ne mentionne d'ailleurs ni l'intelligence artificielle ni aucune "
        "technologie numérique, et le raisonnement « peut inclure... si... » est un "
        "raisonnement hypothétique interdit.\n"
        "Réponse correcte : Décision NON, car l'article n'a pas été écrit dans le "
        "but de réguler l'intelligence artificielle et ne la mentionne pas "
        "explicitement.\n\n"

        "Réponds TOUJOURS en deux parties, dans cet ordre exact, sans aucun autre "
        "texte avant, après ou entre les deux :\n"
        "Justification: [une phrase maximum, citant le passage exact du texte qui "
        "montre que le BUT de l'article correspond à la question — pas de "
        "paraphrase spéculative]\n"
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
