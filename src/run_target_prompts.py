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
# Cadrage des prompts : éviter les correctifs lexicaux propres à un domaine
# (ex. "véhicule automobile" != "automatisé"). Le corpus couvre des domaines
# beaucoup trop variés (circulation routière, analyses génétiques, marchés
# publics...) pour qu'un correctif pensé pour un domaine particulier
# généralise aux autres — au mieux inutile, au pire un nouveau piège. La
# structure retenue est donc :
#   1. Un bloc de CONTEXTE partagé, identique pour les 12 cibles, qui
#      explique en détail et avec un vocabulaire riche ce qu'est
#      l'intelligence artificielle et selon quelle logique les États la
#      régulent (Enabling : promouvoir son développement ; Safeguarding :
#      encadrer les risques qu'elle pose). Ce bloc ancre le lexique de l'IA
#      sans jamais s'appuyer sur un domaine d'application particulier.
#   2. Un bloc CIBLE, propre à chaque code, avec une définition précise et
#      des exemples concrets ILLUSTRANT des mesures qui satisferaient
#      réellement cette cible (pas des pièges négatifs d'un domaine tiers).
#   3. Un ANCRAGE par défaut sceptique : le LLM part de la position "à
#      première vue, cet article n'a aucun rapport avec l'IA" et ne doit en
#      changer que sur preuve explicite — ce qui est plus difficile à
#      renverser qu'une question neutre invitant à chercher un lien.
from __future__ import annotations

from collections import OrderedDict

import pandas as pd


def _p(text: str) -> str:
    """Aplatit un bloc triple-quoté multi-lignes en un seul paragraphe.

    Permet d'écrire un texte long comme UNE chaîne continue et lisible dans
    le source, sans le découper en fragments "..." "..." concaténés ligne
    par ligne.
    """
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Bloc de contexte partagé par les 12 cibles : ce qu'est l'IA, et la logique
# Enabling / Safeguarding qui structure la régulation étatique de l'IA. Pas
# de domaine d'application concret ici — uniquement du vocabulaire et des
# concepts liés à l'IA elle-même.
# ---------------------------------------------------------------------------

_AI_CONTEXT = _p("""
    On entend par intelligence artificielle des systèmes informatiques
    capables, à partir de données, d'apprendre (apprentissage automatique /
    machine learning), de raisonner, de faire des prédictions, de prendre
    des décisions ou de produire un résultat (texte, image, son,
    classification, recommandation, prédiction) de façon autonome ou
    semi-autonome — par exemple des modèles de traitement du langage, des
    systèmes de vision par ordinateur, des systèmes de recommandation, des
    systèmes de décision automatisée fondés sur des données, ou des
    systèmes robotiques dotés d'une capacité de décision autonome. Selon la
    cible évaluée (voir sa définition ci-dessous), un système automatisé de
    traitement de données peut aussi être concerné, même sans apprentissage
    automatique à proprement parler — la définition de la cible précise
    exactement ce qui compte pour elle.
""") + " " + _p("""
    Face à cette technologie, un État régule selon deux logiques
    distinctes. D'une part, il cherche à PROMOUVOIR le développement et
    l'adoption de l'intelligence artificielle, parce qu'elle représente une
    opportunité économique et sociale : il finance la recherche, développe
    les compétences nécessaires, facilite l'accès aux données et aux
    capacités de calcul, encourage son adoption par les organisations, et
    facilite son expérimentation et son entrée sur le marché. D'autre part,
    il cherche à ENCADRER les risques concrets que l'intelligence
    artificielle fait peser sur les personnes et la société : atteintes à
    la vie privée, atteintes aux droits de propriété intellectuelle,
    défaillances de sécurité, décisions inexplicables ou incontestables,
    conséquences lourdes pour les droits fondamentaux, ou préjudices
    sociétaux liés à la diffusion automatisée d'information.
""")

# ---------------------------------------------------------------------------
# Les 12 cibles opérationnelles, dans l'ordre du tableau 2.3 du PDF.
# `definition` : ce que doit faire la norme pour satisfaire cette cible.
# `examples` : 2 exemples concrets ILLUSTRANT des mesures qui satisferaient
# réellement cette cible (pas des contre-exemples d'un domaine tiers).
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
                    dispositif concret de soutien à la recherche, à
                    l'expérimentation ou à l'innovation — financement,
                    création d'une infrastructure ou d'un centre de
                    recherche dédié, régime d'autorisation d'essais ou de
                    phase pilote, ou fonction d'évaluation et de conseil sur
                    les risques et opportunités de nouvelles technologies —
                    dont l'objet est spécifiquement l'intelligence
                    artificielle, un système automatisé de traitement de
                    données, ou les nouvelles technologies en général. Un
                    dispositif de soutien à la recherche ou à l'innovation
                    qui vise un autre domaine scientifique ou technique
                    (génétique, médecine, énergie, etc.) sans lien avec
                    l'intelligence artificielle ou l'automatisation ne
                    satisfait pas cette cible.
                """) + " " + _p("""
                    Un article qui se contente d'attribuer un droit, de
                    définir des notions juridiques, d'instituer un organe de
                    surveillance, de médiation ou de traitement de plaintes,
                    ou d'encadrer une activité existante (radiodiffusion,
                    protection des données, surveillance policière,
                    accessibilité des œuvres pour personnes handicapées)
                    n'est PAS un dispositif de soutien à la recherche ou à
                    l'innovation — même s'il emploie incidemment des mots
                    comme « innovation », « technologie » ou « mise à
                    disposition ». Ce rapprochement lexical superficiel ne
                    suffit pas : il faut un mécanisme concret de soutien
                    dont l'objet porte explicitement sur l'intelligence
                    artificielle, un système automatisé, ou les nouvelles
                    technologies. Si le texte ne mentionne ni l'intelligence
                    artificielle, ni un système automatisé ou informatique,
                    ni les nouvelles technologies, la réponse est NON — même
                    si le domaine régulé semble par ailleurs moderne ou
                    technique.
                """),
                "examples": [
                    "Un fonds public finançant des projets de recherche en intelligence artificielle dans les universités.",
                    "La création d'un centre national de recherche dédié à l'intelligence artificielle.",
                    "Une autorisation d'essais pilotes de durée limitée pour des systèmes automatisés ou pour un traitement automatisé de données, assortie d'un suivi et d'un rapport d'évaluation avant leur généralisation.",
                    "Un service spécialisé chargé, à la demande des autorités, d'évaluer les risques et les opportunités liés à l'utilisation de nouvelles technologies, y compris l'intelligence artificielle.",
                ],
            },
        ),
        (
            "SKILLS_HUMAN_CAPITAL",
            {
                "name": "Compétences & Capital humain",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme met en place un
                    dispositif concret de formation, d'enseignement ou de
                    développement de compétences dont le CONTENU pédagogique
                    porte explicitement et spécifiquement sur l'intelligence
                    artificielle, la science des données ou le calcul
                    informatique (par exemple : apprentissage automatique,
                    programmation, analyse de données, infrastructure de
                    calcul) — un programme de formation continue, une
                    filière ou un cursus universitaire, un module de
                    formation professionnelle, ou une bourse d'études, dont
                    l'objet affiché est l'acquisition de ces compétences.
                """) + " " + _p("""
                    Un dispositif de formation, d'éducation ou de
                    qualification professionnelle NE satisfait PAS cette
                    cible dès lors que son contenu ne porte pas sur l'IA, les
                    données ou le calcul — même s'il emploie les mots «
                    formation », « compétences » ou « apprentissage ». C'est
                    notamment le cas : de la formation à la conduite d'un
                    véhicule (permis probatoire, cours de sensibilisation aux
                    dangers de la route) ; de la formation professionnelle
                    continue dans un autre domaine (santé, sécurité,
                    comptabilité, journalisme, etc.) ; des procédures
                    d'agrément, d'habilitation ou de qualification d'un
                    laboratoire, d'une entreprise ou d'un expert (certificat
                    de sécurité, accréditation qualité) ; d'une expertise ou
                    évaluation psychologique ou médicale d'un individu ; ou
                    plus largement de toute exigence de compétence,
                    d'expérience ou de qualification professionnelle
                    mentionnée dans un texte qui ne porte pas sur l'IA. Le
                    seul fait qu'un article organise le fonctionnement d'un
                    organe, définisse un droit ou une notion juridique,
                    encadre une procédure de surveillance ou de traitement
                    de données, ou porte sur des statistiques sectorielles,
                    n'est pas non plus un dispositif de formation ou de
                    développement de compétences, même si le domaine régulé
                    paraît technique ou numérique. Si le texte ne décrit pas
                    une activité de formation, d'enseignement ou
                    d'apprentissage dont l'objet explicite est
                    l'intelligence artificielle, la science des données ou
                    le calcul informatique, la réponse est NON.
                """),
                "examples": [
                    "Un programme de formation continue en science des données et en intelligence artificielle pour des employés de l'administration.",
                    "La création d'une filière universitaire spécialisée en apprentissage automatique.",
                    "Une bourse d'études destinée à financer un cursus de troisième cycle en intelligence artificielle.",
                ],
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
                    d'intelligence artificielle. Une norme qui organise la
                    collecte ou la gestion de données publiques,
                    administratives ou statistiques en général, sans lien
                    avec le développement de systèmes d'intelligence
                    artificielle, ne satisfait pas cette cible.
                """),
                "examples": [
                    "Une obligation faite aux administrations de mettre à disposition des jeux de données ouvertes utilisables pour entraîner des systèmes d'intelligence artificielle.",
                    "Un cadre légal facilitant le partage de données de santé anonymisées à des fins de recherche en intelligence artificielle.",
                ],
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
                "examples": [
                    "Un investissement public dans un centre de calcul destiné à l'entraînement de modèles d'intelligence artificielle.",
                    "Une subvention pour l'achat de puces spécialisées dans le calcul pour l'intelligence artificielle par des PME.",
                ],
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
                "examples": [
                    "Un programme incitant les PME à adopter des outils d'intelligence artificielle dans leur processus de production.",
                    "Une obligation pour les administrations publiques d'intégrer des outils d'intelligence artificielle dans le traitement de certains dossiers.",
                ],
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
                "examples": [
                    "Un bac à sable réglementaire permettant de tester un système d'intelligence artificielle médicale avant sa mise sur le marché.",
                    "Une procédure d'autorisation temporaire simplifiée pour tester un dispositif fondé sur l'intelligence artificielle.",
                ],
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
                    artificielle OU par un système automatisé de traitement
                    de données (logiciel, système informatique de
                    traitement de données à caractère personnel). Une norme
                    de protection des données qui ne vise pas spécifiquement
                    un tel traitement — par exemple une règle sur la
                    collecte ou la conservation de documents papier, ou sur
                    une procédure purement administrative sans système
                    informatique — ne satisfait pas cette cible.
                """),
                "examples": [
                    "Une obligation d'anonymiser les données utilisées pour entraîner un système d'intelligence artificielle avant leur réutilisation.",
                    "Un droit pour toute personne de connaître les données ayant servi à entraîner un modèle d'intelligence artificielle qui la concerne.",
                    "Une procédure de certification indépendante pour les systèmes ou logiciels de traitement de données personnelles.",
                ],
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
                    l'intelligence artificielle ou un système de traitement
                    automatisé est explicite : données d'entraînement,
                    œuvres protégées utilisées pour entraîner un système
                    d'intelligence artificielle, ou contenu généré par un
                    système d'intelligence artificielle ou par un traitement
                    automatisé.
                """),
                "examples": [
                    "Une règle attribuant les droits d'auteur sur un contenu généré par un système d'intelligence artificielle.",
                    "Une obligation de rémunérer les titulaires de droits dont les œuvres ont servi de données d'entraînement à un modèle d'intelligence artificielle.",
                ],
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
                    artificielle OU des systèmes automatisés de traitement
                    de données. Une exigence de cybersécurité générale, sans
                    lien avec un système informatique de traitement de
                    données (par exemple une exigence de sécurité physique
                    d'un bâtiment ou d'un coffre-fort), ne satisfait pas
                    cette cible.
                """),
                "examples": [
                    "Une exigence de test de robustesse avant la mise en service d'un système d'intelligence artificielle utilisé dans un contexte critique.",
                    "Une obligation de certification de sécurité pour les systèmes d'intelligence artificielle utilisés dans une infrastructure critique.",
                    "Une obligation de certification de sécurité pour les systèmes automatisés de traitement de données personnelles.",
                ],
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
                    issue d'un système d'intelligence artificielle ou d'un
                    traitement automatisé de données.
                """),
                "examples": [
                    "Une obligation d'informer une personne qu'une décision la concernant a été prise par un système d'intelligence artificielle.",
                    "Un droit de recours contre une décision administrative rendue à l'aide d'un système d'intelligence artificielle.",
                ],
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
                "examples": [
                    "Un encadrement spécifique de l'utilisation de systèmes d'intelligence artificielle pour évaluer l'éligibilité à des prestations sociales.",
                    "Une interdiction d'utiliser un système d'intelligence artificielle pour prendre seul une décision de refus de crédit.",
                ],
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
                    d'information par un système d'intelligence
                    artificielle ou par un système automatisé. Une
                    régulation générale des médias ou de la désinformation,
                    sans lien structurel avec l'automatisation ou
                    l'intelligence artificielle, ne satisfait pas cette
                    cible.
                """),
                "examples": [
                    "Une obligation d'étiqueter le contenu généré ou modifié par intelligence artificielle (deepfake) diffusé publiquement.",
                    "Une règle limitant l'usage de systèmes de recommandation automatisés qui amplifient la désinformation.",
                ],
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
    examples = "\n".join(f"- {ex}" for ex in d["examples"])

    blocks = [
        "Tu es un expert en analyse des politiques publiques et du droit suisse.",
        f"## Contexte : l'intelligence artificielle et sa régulation\n\n{_AI_CONTEXT}",
        f"## Cible à évaluer : {name}\n\n{d['definition']}\n\n"
        f"Exemples de mesures qui satisferaient cette cible :\n{examples}",
        "## Calibration\n\n"
        "La plupart des articles de loi n'ont aucun rapport avec l'intelligence "
        "artificielle : ce sont des articles sur la circulation routière, la "
        "fiscalité, l'état civil, les marchés publics, la santé, etc. Pour ces "
        "articles-là, NON est la bonne réponse, et ce n'est pas un échec de répondre "
        "NON. Mais évalue chaque article sur ses propres mérites, à partir de la "
        "définition et des exemples ci-dessus : si le texte correspond clairement à "
        "la cible, réponds OUI, même si ce cas est rare. Ne rejette pas un article "
        "juste parce que la plupart des articles sont à rejeter.",
        "Réponds TOUJOURS en deux parties, dans cet ordre exact, sans aucun autre "
        "texte avant, après ou entre les deux :\n"
        "Justification: [une phrase maximum]\n"
        "Décision: NON ou OUI\n\n"
        'La ligne "Décision:" est OBLIGATOIRE et doit toujours être présente.',
    ]
    return "\n\n".join(blocks)


USER_TEMPLATE = """Texte :
{article_text}

Réponds à la question posée dans tes instructions.

Réponds en deux parties dans cet ordre exact :
Justification: [une phrase maximum]
Décision: NON ou OUI"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
