# src/run_inst_prompts.py
# Un prompt système dédié par instrument (une question binaire par instrument, au lieu
# d'une seule question générique "contient-il un instrument ?" comme dans run1_prompts.py).
# La taxonomie (les 7 codes) reste définie une seule fois dans run5_prompts.py.
from __future__ import annotations

import pandas as pd

from src.run5_prompts import INSTRUMENT_CODES

# ---------------------------------------------------------------------------
# Définitions détaillées par instrument : nom lisible, définition, critères
# d'inclusion/exclusion (avec distinction explicite des autres instruments,
# pour éviter que le LLM confonde des mécanismes voisins), et exemples.
# ---------------------------------------------------------------------------

INSTRUMENT_DEFINITIONS: "dict[str, dict]" = {
    "VOLUNTARY": {
        "name": "instruments volontaires",
        "definition": (
            "Action de l'État qui encourage, facilite ou coordonne un comportement sans le "
            "rendre juridiquement contraignant. L'acteur reste libre de suivre ou non la "
            "démarche proposée : il n'existe ni obligation, ni interdiction, ni conséquence "
            "juridique directe en cas de non-participation."
        ),
        "include": [
            "campagnes de sensibilisation ou d'information menées par une autorité publique ;",
            "programmes d'encouragement, de promotion ou de coordination volontaire entre "
            "acteurs, sans obligation légale de participer ;",
            "recommandations, lignes directrices, chartes ou labels dont l'adhésion est "
            "facultative ;",
            "mise à disposition volontaire d'outils, de plateformes, de conseils ou d'un "
            "accompagnement, sans obligation d'y recourir.",
        ],
        "exclude": [
            "toute mesure qui impose une obligation contraignante à un acteur relève de "
            "OBLIGATION, pas de VOLUNTARY ;",
            "tout mécanisme financier (subvention, aide, taxe, exonération) relève de "
            "TAXES_SUBSIDIES, même s'il vise à encourager un comportement ;",
            "toute interdiction relève de PROHIBITION_BAN ;",
            "la simple création ou organisation d'une autorité sans pouvoir ni programme "
            "d'encouragement concret n'est pas un instrument.",
        ],
        "examples_oui": [
            (
                "La Confédération peut soutenir des campagnes d'information visant à "
                "sensibiliser le public aux risques liés à l'utilisation de l'intelligence "
                "artificielle.",
                "action de sensibilisation menée par l'État, sans obligation pour qui que "
                "ce soit.",
            ),
            (
                "Les cantons encouragent la coordination volontaire entre acteurs privés pour "
                "l'élaboration de bonnes pratiques en matière de protection des données.",
                "encouragement à une démarche facultative, sans contrainte légale.",
            ),
        ],
        "examples_non": [
            (
                "Les exploitants doivent informer les autorités de tout incident grave dans un "
                "délai de 24 heures.",
                "obligation contraignante de notification (OBLIGATION), pas une démarche "
                "volontaire.",
            ),
            (
                "La présente loi a pour but de promouvoir l'innovation numérique.",
                "énoncé de but général sans mécanisme concret d'encouragement.",
            ),
        ],
    },
    "TAXES_SUBSIDIES": {
        "name": "taxes et subventions",
        "definition": (
            "Mécanisme financier par lequel l'État prélève (taxe, impôt, redevance, "
            "contribution) ou verse (subvention, aide, allocation, exonération) des ressources "
            "monétaires à des acteurs privés ou publics, afin d'influencer un comportement, de "
            "compenser une externalité ou de soutenir une activité."
        ),
        "include": [
            "création ou modification d'une taxe, redevance, contribution ou d'un impôt "
            "spécifique lié à une activité ou un secteur ;",
            "subventions, aides financières, allocations ou prestations versées à des acteurs "
            "privés ou publics ;",
            "exonérations fiscales, réductions de charges ou crédits d'impôt conditionnés à un "
            "comportement ;",
            "primes ou incitations financières directes.",
        ],
        "exclude": [
            "le financement d'infrastructures, d'équipements ou de recherche directement pris "
            "en charge et exploité par l'État relève de PUBLIC_INVESTMENT, pas de "
            "TAXES_SUBSIDIES ;",
            "les obligations non financières (documentation, notification, conformité) "
            "relèvent de OBLIGATION ;",
            "les sanctions pécuniaires punitives infligées pour non-respect d'une règle "
            "(amendes) relèvent de LIABILITY, pas d'une taxe : une taxe est un prélèvement "
            "régulier lié à une activité légale, une amende sanctionne une infraction.",
        ],
        "examples_oui": [
            (
                "Une redevance annuelle est perçue sur les centres de données grands "
                "consommateurs d'énergie.",
                "création d'une redevance liée à une activité.",
            ),
            (
                "Des subventions peuvent être octroyées aux entreprises qui investissent dans "
                "la formation de leur personnel aux outils numériques.",
                "versement d'une aide financière à des acteurs privés.",
            ),
        ],
        "examples_non": [
            (
                "Quiconque contrevient à l'art. 12 est puni d'une amende de 10 000 francs au "
                "plus.",
                "sanction pécuniaire punitive (LIABILITY), pas une taxe.",
            ),
            (
                "La Confédération finance la construction d'un centre de calcul public destiné "
                "à la recherche.",
                "investissement public direct (PUBLIC_INVESTMENT), pas un transfert financier "
                "à un tiers.",
            ),
        ],
    },
    "PUBLIC_INVESTMENT": {
        "name": "investissement et marchés publics",
        "definition": (
            "L'État engage et gère lui-même des ressources publiques — infrastructures, "
            "équipements, capacités, marchés publics, achats — plutôt que de transférer de "
            "l'argent à des tiers. Le maître d'ouvrage ou l'acheteur est l'État lui-même."
        ),
        "include": [
            "investissements publics dans des infrastructures, équipements ou capacités "
            "(calcul, énergie, recherche, formation) exploités par ou pour l'État ;",
            "marchés publics, appels d'offres ou achats de biens et services par une autorité ;",
            "création ou financement direct d'institutions, d'instituts ou d'infrastructures "
            "publiques.",
        ],
        "exclude": [
            "le versement d'argent à des acteurs privés sous forme de subvention ou d'aide "
            "relève de TAXES_SUBSIDIES, même si l'objectif est similaire ;",
            "la simple organisation interne d'une autorité (compétence, structure interne) "
            "sans investissement concret n'est pas un instrument ;",
            "la réglementation de l'accès à une ressource sans investissement direct de l'État "
            "relève plutôt de OBLIGATION ou PLANNING_EVALUATION selon le mécanisme.",
        ],
        "examples_oui": [
            (
                "La Confédération investit dans la construction d'infrastructures de calcul à "
                "haute performance destinées à la recherche publique.",
                "investissement public direct dans une infrastructure.",
            ),
            (
                "Les marchés publics de la Confédération intègrent des critères d'acquisition "
                "de systèmes d'intelligence artificielle sécurisés.",
                "mécanisme de marché public.",
            ),
        ],
        "examples_non": [
            (
                "Les entreprises actives dans le secteur numérique peuvent bénéficier d'une "
                "aide financière pour leurs investissements.",
                "subvention versée à des tiers (TAXES_SUBSIDIES), pas un investissement direct "
                "de l'État.",
            ),
            (
                "L'Office fédéral de la statistique est chargé de la coordination interne des "
                "données numériques de l'administration.",
                "organisation interne d'une autorité, sans investissement concret.",
            ),
        ],
    },
    "PROHIBITION_BAN": {
        "name": "interdiction",
        "definition": (
            "Interdiction directe d'une activité, d'un comportement, d'un produit ou d'une "
            "technologie — totale ou conditionnelle (« ne peut... que si ») —, ou droit "
            "exclusif légal créant une interdiction implicite pour les tiers."
        ),
        "include": [
            "formulations « il est interdit de », « nul ne peut », « ne peut pas », « ne "
            "peut... que si » (interdiction assortie d'une exception) ;",
            "interdictions totales ou partielles d'une pratique, d'une technologie ou d'un "
            "usage ;",
            "droits exclusifs légaux (monopoles, exclusivités) créant une interdiction "
            "implicite pour les tiers.",
        ],
        "exclude": [
            "une obligation positive d'agir (mettre en œuvre, garantir, documenter) relève de "
            "OBLIGATION, pas d'une interdiction, même si son non-respect est indirectement "
            "sanctionné ;",
            "la sanction prévue en cas de violation d'une interdiction (amende, retrait de "
            "permis) relève de LIABILITY : code ici uniquement l'interdiction elle-même, pas la "
            "conséquence de sa violation ;",
            "une simple condition procédurale d'obtention d'une autorisation, sans énoncé "
            "explicite d'interdiction de principe, relève de OBLIGATION.",
        ],
        "examples_oui": [
            (
                "Il est interdit d'utiliser des systèmes d'identification biométrique à "
                "distance dans les lieux accessibles au public.",
                "interdiction directe et explicite.",
            ),
            (
                "Nul ne peut mettre sur le marché un système d'intelligence artificielle à "
                "haut risque sans certification préalable.",
                "interdiction conditionnelle (« ne peut... que si »).",
            ),
        ],
        "examples_non": [
            (
                "Les exploitants doivent mettre en œuvre des mesures appropriées pour prévenir "
                "le risque identifié.",
                "obligation positive (OBLIGATION), pas une interdiction.",
            ),
            (
                "Quiconque contrevient à l'interdiction prévue à l'art. 8 est puni d'une "
                "amende.",
                "sanction associée à une interdiction (LIABILITY), pas l'interdiction "
                "elle-même.",
            ),
        ],
    },
    "PLANNING_EVALUATION": {
        "name": "planification et évaluation",
        "definition": (
            "Élaboration de plans ou de stratégies, évaluations et audits obligatoires, "
            "rapports, registres ou inventaires officiels, bacs à sable réglementaires, "
            "ou obligations d'évaluation et de gestion des risques imposées à des "
            "organisations. Le mécanisme central est un exercice de planification, "
            "d'évaluation ou de suivi — même lorsqu'il est rendu obligatoire."
        ),
        "include": [
            "élaboration de plans, stratégies ou programmes officiels par une autorité ;",
            "évaluations, audits ou analyses de risques obligatoires imposés à des "
            "organisations ;",
            "rapports périodiques, registres, inventaires ou bases de données officielles ;",
            "bacs à sable réglementaires (sandboxes), projets pilotes encadrés ;",
            "suivi, monitoring ou évaluation d'une politique publique.",
        ],
        "exclude": [
            "une simple obligation de documentation technique liée à un produit (notice, "
            "manuel d'utilisation) sans dimension d'évaluation ou de planification relève de "
            "OBLIGATION ;",
            "une obligation de conservation de données sans exercice d'évaluation ou d'analyse "
            "relève de OBLIGATION ;",
            "une interdiction ou une obligation de comportement sans dimension planificatrice "
            "relève de PROHIBITION_BAN ou OBLIGATION.",
        ],
        "examples_oui": [
            (
                "Le Conseil fédéral établit tous les quatre ans un rapport sur l'état du "
                "développement de l'intelligence artificielle en Suisse.",
                "rapport périodique officiel.",
            ),
            (
                "Les exploitants de systèmes à haut risque doivent réaliser une analyse "
                "d'impact avant leur mise en service.",
                "évaluation obligatoire des risques.",
            ),
        ],
        "examples_non": [
            (
                "Les exploitants doivent conserver les données de fonctionnement du système "
                "pendant cinq ans.",
                "obligation de conservation simple (OBLIGATION), pas un exercice d'évaluation.",
            ),
            (
                "L'utilisation de systèmes de notation sociale par les autorités est "
                "interdite.",
                "interdiction (PROHIBITION_BAN), pas une planification ou une évaluation.",
            ),
        ],
    },
    "OBLIGATION": {
        "name": "obligation",
        "definition": (
            "Obligation positive imposée directement à des acteurs privés ou à des "
            "opérateurs : règle de comportement obligatoire, prescription technique, exigence "
            "de documentation ou d'enregistrement, condition à remplir pour obtenir ou "
            "conserver une autorisation, obligation de signalement ou de notification."
        ),
        "include": [
            "obligations positives de faire quelque chose (mettre en œuvre, garantir, "
            "assurer, documenter) imposées à un acteur réglementé ;",
            "prescriptions techniques ou normes de conformité obligatoires ;",
            "obligations de notification, de signalement ou d'information envers une "
            "autorité ;",
            "conditions à remplir pour obtenir ou conserver une autorisation ou un permis ;",
            "exigences de documentation, d'enregistrement ou de tenue de registre directement "
            "liées à un produit ou une activité (sans dimension de planification stratégique).",
        ],
        "exclude": [
            "une interdiction pure (« il est interdit de », « nul ne peut ») relève de "
            "PROHIBITION_BAN, pas d'une obligation positive ;",
            "un exercice de planification, d'évaluation ou de suivi stratégique (rapport "
            "périodique, analyse d'impact, registre officiel) relève de "
            "PLANNING_EVALUATION ;",
            "un mécanisme financier (taxe, subvention) relève de TAXES_SUBSIDIES ;",
            "une sanction pour non-respect d'une obligation relève de LIABILITY : code ici "
            "uniquement l'obligation elle-même, pas sa sanction ;",
            "une règle purement procédurale régissant le déroulement d'une procédure "
            "administrative ou judiciaire n'est pas un instrument.",
        ],
        "examples_oui": [
            (
                "Les exploitants doivent garantir la supervision humaine effective des "
                "systèmes à haut risque.",
                "obligation positive imposée à un acteur réglementé.",
            ),
            (
                "Avant sa mise sur le marché, tout système doit être accompagné d'une "
                "documentation technique complète.",
                "exigence documentaire obligatoire.",
            ),
        ],
        "examples_non": [
            (
                "Les exploitants doivent réaliser une analyse d'impact avant la mise en "
                "service du système.",
                "évaluation obligatoire (PLANNING_EVALUATION), pas une simple obligation de "
                "comportement.",
            ),
            (
                "Il est interdit de mettre sur le marché un système non conforme aux exigences "
                "applicables.",
                "interdiction (PROHIBITION_BAN), pas une obligation positive.",
            ),
        ],
    },
    "LIABILITY": {
        "name": "régime de responsabilité",
        "definition": (
            "Mécanisme de responsabilité servant de levier d'application d'une obligation "
            "réglementaire précise : saisie et confiscation, sanctions administratives ou "
            "pécuniaires pour non-respect d'une règle, assurance obligatoire, retrait ou "
            "suspension d'un permis en cas d'infraction. N'inclut PAS la répartition par "
            "défaut de la responsabilité civile entre acteurs privés (détenteur, propriétaire, "
            "tiers, lésé), ni les modalités générales d'indemnisation, qui relèvent du droit "
            "privé."
        ),
        "include": [
            "amendes, sanctions administratives ou pénales pour non-respect d'une règle "
            "réglementaire ;",
            "saisie, confiscation, retrait ou suspension d'un permis ou d'une autorisation en "
            "cas d'infraction ;",
            "assurance obligatoire liée à l'exercice d'une activité réglementée ;",
            "pouvoir d'exécution forcée ou mesures correctives imposées en cas de "
            "non-conformité.",
        ],
        "exclude": [
            "la répartition par défaut de la responsabilité civile entre acteurs privés "
            "(détenteur, propriétaire, tiers lésé) relève du droit privé général, PAS d'un "
            "instrument de politique publique ;",
            "les modalités générales d'indemnisation entre particuliers relèvent du droit "
            "privé ;",
            "l'obligation ou l'interdiction dont la violation est sanctionnée doit être codée "
            "séparément sous OBLIGATION ou PROHIBITION_BAN, pas ici : code ici uniquement le "
            "mécanisme de sanction ou de garantie lui-même.",
        ],
        "examples_oui": [
            (
                "L'autorité compétente peut retirer l'autorisation d'exploitation en cas de "
                "violation grave et répétée des exigences de sécurité.",
                "retrait de permis en cas d'infraction.",
            ),
            (
                "Les exploitants doivent souscrire une assurance couvrant les dommages causés "
                "par le système avant sa mise en service.",
                "assurance obligatoire liée à l'activité réglementée.",
            ),
        ],
        "examples_non": [
            (
                "Le détenteur du système répond du dommage causé, à moins qu'il ne prouve "
                "qu'aucune faute ne lui est imputable.",
                "répartition par défaut de la responsabilité civile entre particuliers, "
                "explicitement exclue.",
            ),
            (
                "Les parties peuvent convenir contractuellement des modalités de réparation du "
                "dommage.",
                "modalité générale d'indemnisation relevant du droit privé.",
            ),
        ],
    },
}

assert set(INSTRUMENT_DEFINITIONS) == set(INSTRUMENT_CODES), (
    "INSTRUMENT_DEFINITIONS doit couvrir exactement les codes de run5_prompts.INSTRUMENT_CODES"
)


def build_system_prompt(code: str) -> str:
    if code not in INSTRUMENT_DEFINITIONS:
        raise KeyError(f"Unknown instrument code: {code}")

    d = INSTRUMENT_DEFINITIONS[code]

    include_bullets = "\n".join(f"- {b}" for b in d["include"])
    exclude_bullets = "\n".join(f"- {b}" for b in d["exclude"])

    examples: list[str] = []
    n = 1
    for text, justif in d["examples_oui"]:
        examples.append(f"{n}. « {text} » → OUI : {justif}")
        n += 1
    for text, justif in d["examples_non"]:
        examples.append(f"{n}. « {text} » → NON : {justif}")
        n += 1
    examples_block = "\n".join(examples)

    return (
        "Tu es un expert en analyse des politiques publiques et du droit suisse.\n\n"
        "## Tâche\n\n"
        f"Détermine si l'article de loi contient spécifiquement l'instrument de politique "
        f"publique suivant : {d['name'].upper()} ({code}).\n\n"
        f"Réponds uniquement à la question suivante : cet article contient-il l'instrument "
        f"« {d['name']} » ?\n\n"
        "Un article de loi suisse est souvent composé de plusieurs alinéas. Examine CHAQUE "
        "alinéa séparément : il suffit qu'UN SEUL alinéa relève de cet instrument précis pour "
        "que l'article entier soit classé OUI, même si les autres alinéas relèvent d'autres "
        "instruments ou n'en contiennent aucun.\n\n"
        "## Définition\n\n"
        f"{d['definition']}\n\n"
        "## Inclure\n\n"
        f"Classe l'article OUI lorsqu'il établit, modifie ou autorise lui-même au moins un "
        f"mécanisme correspondant à {d['name']}, notamment :\n"
        f"{include_bullets}\n\n"
        "Ces exemples sont illustratifs, pas exhaustifs. Détermine si la disposition remplit la "
        "même fonction même si son mécanisme précis n'est pas listé ci-dessus.\n\n"
        "## Exclure\n\n"
        "Ne classe PAS l'article ici lorsque le mécanisme qu'il contient relève en réalité d'un "
        "AUTRE instrument de politique publique, ou d'aucun instrument. En particulier :\n"
        f"{exclude_bullets}\n\n"
        "## Exemples\n\n"
        f"{examples_block}\n\n"
        "## Règle de décision\n\n"
        f"Applique le test suivant : l'article établit-il, modifie-t-il ou autorise-t-il "
        f"lui-même un mécanisme correspondant précisément à {d['name']}, tel que défini "
        "ci-dessus ?\n"
        "- Si l'article contient plusieurs alinéas ou dispositions, classe-le OUI dès qu'AU "
        "MOINS UN d'entre eux établit, modifie ou autorise un tel mécanisme. Ne fais JAMAIS la "
        "moyenne ni le vote majoritaire entre alinéas.\n"
        "- Si le mécanisme identifié correspond mieux à un AUTRE instrument (voir section "
        "\"Exclure\" ci-dessus), réponds NON : ne code jamais un même mécanisme sous plusieurs "
        "instruments à la fois.\n"
        "- N'infère pas la présence de cet instrument à partir de l'objectif général de la loi, "
        "du titre de l'article, du secteur de politique publique, ou de mécanismes qui "
        "pourraient exister ailleurs dans la législation.\n"
        "- Fonde la décision uniquement sur le contenu de l'article fourni.\n"
        "- En cas d'incertitude, privilégie NON, sauf si un mécanisme correspondant précisément "
        "à cet instrument peut être identifié explicitement dans l'article lui-même.\n\n"
        "Réponds TOUJOURS en deux parties, dans cet ordre exact, sans aucun autre texte avant, "
        "après ou entre les deux :\n"
        "Justification: [1 à 2 phrases maximum, ancrées dans le texte]\n"
        "Décision: OUI ou NON\n\n"
        "La ligne \"Décision:\" est OBLIGATOIRE et doit toujours être présente, même si tu "
        "hésites : dans ce cas, tranche selon la règle de décision ci-dessus plutôt que "
        "d'omettre la ligne."
    )


USER_TEMPLATE = """Texte :
{article_text}

Cet article contient-il l'instrument de politique publique décrit dans les instructions ?

Réponds en deux parties dans cet ordre exact :
Justification: [1 à 2 phrases maximum]
Décision: OUI ou NON"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
