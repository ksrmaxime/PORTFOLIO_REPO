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
#
# Ces définitions ont été resserrées après un premier essai qui produisait
# trop de faux positifs (VOLUNTARY, TAXES_SUBSIDIES et PLANNING_EVALUATION
# étaient définis de façon trop large ; LIABILITY était défini comme des
# sanctions/enforcement au lieu d'une allocation de responsabilité, ce qui
# contredisait la définition du PoC). Voir aussi les notes "Exclure" qui
# précisent, catégorie par catégorie, ce qui ressemble à l'instrument sans en
# être un.
# ---------------------------------------------------------------------------

INSTRUMENT_DEFINITIONS: "dict[str, dict]" = {
    "VOLUNTARY": {
        "name": "instruments volontaires",
        "definition": (
            "Mécanisme FORMALISÉ de régulation non contraignante auquel des acteurs publics "
            "et/ou privés sont invités à adhérer ou à se conformer volontairement : code de "
            "conduite, charte, engagement volontaire, accord volontaire entre l'État et des "
            "acteurs privés, standard ou label non contraignant utilisé comme mécanisme de "
            "pilotage. La simple communication, information ou sensibilisation du public par "
            "une autorité n'est PAS un instrument volontaire : il ne s'agit pas d'un mécanisme "
            "auquel un acteur adhère ou se conforme, seulement d'un message diffusé."
        ),
        "include": [
            "codes de conduite élaborés ou reconnus par une autorité, auxquels des acteurs "
            "peuvent adhérer ;",
            "chartes ou engagements volontaires formalisés, pris par des acteurs privés ou "
            "publics ;",
            "accords volontaires conclus entre une autorité et des acteurs privés (ex. "
            "engagements sectoriels négociés) ;",
            "standards, labels ou certifications réellement facultatifs, utilisés comme "
            "mécanisme d'orientation d'un comportement (l'acteur choisit librement d'y "
            "souscrire ou non, sans conséquence juridique en cas de non-adhésion).",
        ],
        "exclude": [
            "une simple campagne d'information, de sensibilisation, de conseil ou "
            "d'accompagnement, sans mécanisme formalisé d'adhésion, N'EST PAS un instrument de "
            "ce portfolio (aucun des 7 codes) : il n'y a rien à quoi l'acteur adhère ou se "
            "conforme ;",
            "toute mesure qui impose une obligation contraignante à un acteur relève de "
            "OBLIGATION, pas de VOLUNTARY ;",
            "tout mécanisme financier (subvention, aide, taxe, exonération) relève de "
            "TAXES_SUBSIDIES, même s'il vise à encourager un comportement ;",
            "toute interdiction relève de PROHIBITION_BAN ;",
            "la simple création ou organisation d'une autorité, sans code, charte, accord ou "
            "label concret, n'est pas un instrument.",
        ],
        "examples_oui": [
            (
                "Les fournisseurs de systèmes d'intelligence artificielle peuvent adhérer à un "
                "code de conduite, élaboré en collaboration avec les associations "
                "professionnelles, définissant des engagements volontaires en matière de "
                "transparence.",
                "code de conduite formalisé, adhésion volontaire.",
            ),
            (
                "Les acteurs du secteur numérique peuvent conclure avec la Confédération des "
                "accords volontaires portant sur la réduction de la consommation énergétique "
                "des centres de données.",
                "accord volontaire formalisé entre l'État et des acteurs privés.",
            ),
        ],
        "examples_non": [
            (
                "La Confédération peut soutenir des campagnes d'information visant à "
                "sensibiliser le public aux risques liés à l'utilisation de l'intelligence "
                "artificielle.",
                "simple information/sensibilisation, sans mécanisme formalisé d'adhésion : ce "
                "n'est pas un instrument de ce portfolio.",
            ),
            (
                "Les exploitants doivent informer les autorités de tout incident grave dans un "
                "délai de 24 heures.",
                "obligation contraignante de notification (OBLIGATION), pas une démarche "
                "volontaire.",
            ),
            (
                "La présente loi a pour but de promouvoir l'innovation numérique.",
                "énoncé de but général sans mécanisme concret.",
            ),
        ],
    },
    "TAXES_SUBSIDIES": {
        "name": "taxes et subventions",
        "definition": (
            "Incitation ou désincitation FISCALE OU FINANCIÈRE utilisée par l'État comme "
            "mécanisme pour orienter (encourager ou décourager) un comportement, un "
            "investissement ou un choix de conformité — pas un simple financement du "
            "fonctionnement de l'État, de ses procédures ou de ses institutions. Le transfert "
            "financier doit lui-même constituer le mécanisme de pilotage, pas un effet "
            "secondaire administratif."
        ),
        "include": [
            "taxes, redevances ou impôts spécifiquement incitatifs, dont la fonction est "
            "d'orienter un comportement (ex. taxe carbone, taxe incitative sur une "
            "consommation) ;",
            "subventions, aides financières ou allocations versées pour encourager un "
            "comportement, un investissement ou une pratique donnée ;",
            "exonérations fiscales, réductions de charges ou crédits d'impôt conditionnés à un "
            "comportement.",
        ],
        "exclude": [
            "les émoluments administratifs couvrant le coût d'une procédure (ex. frais de "
            "traitement d'une demande d'autorisation) ne sont pas de la taxation incitative ;",
            "les primes d'assurance obligatoire relèvent de LIABILITY (garantie d'une "
            "responsabilité), pas de TAXES_SUBSIDIES ;",
            "les contributions de financement institutionnel ou de fonctionnement d'un "
            "organisme, le remboursement de frais et les dommages-intérêts ne sont pas des "
            "instruments fiscaux de pilotage ;",
            "les amendes et sanctions pécuniaires punitives infligées pour non-respect d'une "
            "règle ne sont pas des taxes : une taxe est un prélèvement incitatif régulier lié à "
            "une activité légale, une amende sanctionne une infraction et ne relève d'aucun des "
            "7 instruments de ce portfolio ;",
            "le financement direct d'infrastructures ou d'équipements exploités par l'État "
            "lui-même relève de PUBLIC_INVESTMENT, pas de TAXES_SUBSIDIES.",
        ],
        "examples_oui": [
            (
                "Une taxe incitative est perçue sur les centres de données à forte "
                "consommation énergétique, dont le produit est redistribué aux exploitants qui "
                "réduisent leur consommation.",
                "taxe explicitement conçue pour orienter un comportement.",
            ),
            (
                "Des subventions peuvent être octroyées aux entreprises qui investissent dans "
                "la formation de leur personnel aux outils numériques.",
                "versement d'une aide financière destinée à encourager un comportement.",
            ),
        ],
        "examples_non": [
            (
                "Un émolument est perçu pour le traitement de la demande d'autorisation "
                "d'exploitation.",
                "émolument administratif couvrant un coût de procédure, pas un mécanisme "
                "d'orientation d'un comportement.",
            ),
            (
                "Quiconque contrevient à l'art. 12 est puni d'une amende de 10 000 francs au "
                "plus.",
                "sanction pécuniaire punitive, pas une taxe.",
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
            "Investissement public destiné à CRÉER OU ACCROÎTRE UNE CAPACITÉ SUBSTANTIELLE — "
            "infrastructure, équipement, programme, ressource ou service — financée et "
            "exploitée par l'État lui-même ; ou acquisition publique (marché public) "
            "explicitement utilisée comme mécanisme de politique publique. Le simple "
            "financement du fonctionnement ordinaire d'une administration (budget courant, "
            "salaires, frais de fonctionnement) n'est pas un investissement au sens de cette "
            "catégorie."
        ),
        "include": [
            "investissement public créant ou accroissant une capacité substantielle "
            "(infrastructure de calcul, équipement, programme de recherche, service public) "
            "exploitée par ou pour l'État ;",
            "marchés publics, appels d'offres ou achats de biens et services par une autorité, "
            "utilisés explicitement comme mécanisme de politique publique (ex. critères "
            "d'acquisition orientant le marché).",
        ],
        "exclude": [
            "le versement d'argent à des acteurs privés sous forme de subvention ou d'aide "
            "relève de TAXES_SUBSIDIES, même si l'objectif est similaire ;",
            "le simple financement du fonctionnement administratif d'un organisme (budget de "
            "fonctionnement, salaires, frais courants), sans création de capacité nouvelle, "
            "n'est pas un instrument ;",
            "la simple organisation interne d'une autorité (compétence, structure interne) "
            "sans investissement concret n'est pas un instrument ;",
            "la réglementation de l'accès à une ressource sans investissement direct de l'État "
            "relève plutôt de OBLIGATION ou PLANNING_EVALUATION selon le mécanisme.",
        ],
        "examples_oui": [
            (
                "La Confédération investit dans la construction d'infrastructures de calcul à "
                "haute performance destinées à la recherche publique.",
                "investissement public créant une capacité nouvelle.",
            ),
            (
                "Les marchés publics de la Confédération intègrent des critères d'acquisition "
                "de systèmes d'intelligence artificielle sécurisés, utilisés pour orienter "
                "l'offre du marché.",
                "marché public utilisé comme mécanisme de politique publique.",
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
                "données numériques de l'administration et dispose d'un budget de "
                "fonctionnement annuel.",
                "financement du fonctionnement ordinaire d'une administration, pas création "
                "d'une capacité nouvelle.",
            ),
        ],
    },
    "PROHIBITION_BAN": {
        "name": "interdiction",
        "definition": (
            "Une conduite ou activité identifiable est-elle DIRECTEMENT déclarée interdite, "
            "illicite ou juridiquement impossible ? Il s'agit du mécanisme le plus textuel : "
            "l'article doit dire, en substance, qu'une activité ne peut pas avoir lieu — pas "
            "qu'elle est soumise à des conditions pour avoir lieu."
        ),
        "include": [
            "formulations « il est interdit de », « nul ne peut », « ne peut pas » énonçant une "
            "interdiction directe ;",
            "interdictions totales ou partielles, explicites, d'une pratique, d'une "
            "technologie ou d'un usage.",
        ],
        "exclude": [
            "une formulation « ne peut... que si » (ou équivalente) qui pose une CONDITION "
            "d'accès à une activité par ailleurs autorisée relève d'une obligation/condition "
            "d'accès (OBLIGATION), PAS d'une interdiction : l'activité reste possible si la "
            "condition est remplie ;",
            "un droit exclusif légal (monopole, exclusivité) n'est pas à coder ici, sauf si le "
            "texte formule lui-même une interdiction explicite envers les tiers ;",
            "une obligation positive d'agir (mettre en œuvre, garantir, documenter) relève de "
            "OBLIGATION, pas d'une interdiction, même si son non-respect est indirectement "
            "sanctionné ;",
            "la sanction prévue en cas de violation d'une interdiction (amende, retrait de "
            "permis) est un mécanisme d'exécution distinct, qui ne relève d'aucun des 7 "
            "instruments de ce portfolio : code ici uniquement l'interdiction elle-même, pas sa "
            "sanction.",
        ],
        "examples_oui": [
            (
                "Il est interdit d'utiliser des systèmes d'identification biométrique à "
                "distance dans les lieux accessibles au public.",
                "interdiction directe et explicite.",
            ),
            (
                "La production, l'importation et la mise sur le marché de systèmes de notation "
                "sociale par les autorités publiques sont interdites.",
                "interdiction directe et explicite d'une pratique.",
            ),
        ],
        "examples_non": [
            (
                "Un système d'intelligence artificielle à haut risque ne peut être mis sur le "
                "marché que s'il a été certifié conforme aux exigences applicables.",
                "condition d'accès à une activité par ailleurs autorisée (OBLIGATION), pas une "
                "interdiction directe.",
            ),
            (
                "Les exploitants doivent mettre en œuvre des mesures appropriées pour prévenir "
                "le risque identifié.",
                "obligation positive (OBLIGATION), pas une interdiction.",
            ),
            (
                "Quiconque contrevient à l'interdiction prévue à l'art. 8 est puni d'une "
                "amende.",
                "sanction associée à une interdiction, pas l'interdiction elle-même.",
            ),
        ],
    },
    "PLANNING_EVALUATION": {
        "name": "planification et évaluation",
        "definition": (
            "Mécanisme qui produit SYSTÉMATIQUEMENT une évaluation, une expérimentation, une "
            "planification ou une surveillance STRUCTURÉE, destinée à éclairer, contrôler ou "
            "adapter une politique, une activité ou un système (plan stratégique, audit, "
            "analyse d'impact, évaluation périodique, bac à sable réglementaire, programme "
            "pilote, dispositif de monitoring). Un simple registre, une base de données, un "
            "inventaire administratif ou une obligation ponctuelle de rapport n'en font PAS "
            "partie, sauf s'ils constituent eux-mêmes un exercice d'évaluation structuré."
        ),
        "include": [
            "élaboration d'un plan, d'une stratégie ou d'un programme officiel par une "
            "autorité ;",
            "audit obligatoire, analyse ou évaluation d'impact ;",
            "évaluation périodique structurée d'une politique, d'une activité ou d'un système, "
            "destinée à éclairer une décision ou une adaptation ;",
            "bac à sable réglementaire (sandbox), projet pilote encadré ;",
            "dispositif de monitoring structuré, avec objectif explicite de suivi et "
            "d'ajustement.",
        ],
        "exclude": [
            "un simple registre, inventaire ou base de données administrative n'est pas, en "
            "soi, un exercice d'évaluation ou de planification ; s'il est imposé comme une "
            "exigence de tenue de registre, il relève plutôt de OBLIGATION ;",
            "une obligation de transmettre un rapport ponctuel ne relève de cette catégorie que "
            "si le rapport constitue lui-même une évaluation structurée (et non une simple "
            "transmission d'informations) ; sinon elle relève de OBLIGATION ;",
            "une collecte statistique ou administrative de routine n'est pas, par défaut, un "
            "instrument de cette catégorie ;",
            "une interdiction ou une obligation de comportement sans dimension planificatrice "
            "relève de PROHIBITION_BAN ou OBLIGATION.",
        ],
        "examples_oui": [
            (
                "Le Conseil fédéral établit tous les quatre ans un rapport d'évaluation sur "
                "l'état du développement de l'intelligence artificielle en Suisse, destiné à "
                "orienter la politique en la matière.",
                "évaluation périodique structurée destinée à éclairer une politique publique.",
            ),
            (
                "Les exploitants de systèmes à haut risque doivent réaliser une analyse "
                "d'impact avant leur mise en service.",
                "évaluation obligatoire des risques.",
            ),
        ],
        "examples_non": [
            (
                "Les exploitants tiennent un registre des systèmes qu'ils utilisent.",
                "simple registre administratif, pas un exercice d'évaluation (relève de "
                "OBLIGATION).",
            ),
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
            "Obligation positive imposée directement à un acteur réglementé, PUBLIC OU PRIVÉ "
            "(opérateur, entreprise, mais aussi administration, canton, institution publique) "
            ": règle de comportement obligatoire, prescription technique, exigence de "
            "documentation ou d'enregistrement, obligation de signalement ou de notification. "
            "Une condition juridiquement obligatoire à satisfaire pour pouvoir exercer une "
            "activité compte comme une obligation, même si elle est formulée comme condition "
            "d'autorisation ou de permis (ex. « ne peut... que si »)."
        ),
        "include": [
            "obligations positives de faire quelque chose (mettre en œuvre, garantir, "
            "assurer, documenter) imposées à un acteur réglementé, public ou privé ;",
            "prescriptions techniques ou normes de conformité obligatoires ;",
            "obligations de notification, de signalement ou d'information envers une "
            "autorité ;",
            "conditions substantielles à remplir pour pouvoir exercer une activité, obtenir ou "
            "conserver une autorisation ou un permis (y compris formulées « ne peut... que "
            "si ») ;",
            "exigences de documentation, d'enregistrement ou de tenue de registre directement "
            "liées à un produit ou une activité (sans dimension de planification stratégique).",
        ],
        "exclude": [
            "une interdiction pure, sans condition qui permettrait de rendre l'activité licite "
            "(« il est interdit de », « nul ne peut »), relève de PROHIBITION_BAN, pas d'une "
            "obligation positive ;",
            "un exercice de planification, d'évaluation ou de suivi stratégique (rapport "
            "périodique structuré, analyse d'impact, plan) relève de PLANNING_EVALUATION ;",
            "un mécanisme financier (taxe, subvention) relève de TAXES_SUBSIDIES ;",
            "une sanction pour non-respect d'une obligation est un mécanisme d'exécution "
            "distinct, qui ne relève d'aucun des 7 instruments de ce portfolio : code ici "
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
                "Un système d'intelligence artificielle à haut risque ne peut être mis sur le "
                "marché que s'il a été certifié conforme aux exigences applicables.",
                "condition substantielle d'accès à une activité, imposée comme obligation.",
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
                "Il est interdit d'utiliser des systèmes d'identification biométrique à "
                "distance dans les lieux accessibles au public.",
                "interdiction pure, sans condition d'accès (PROHIBITION_BAN).",
            ),
        ],
    },
    "LIABILITY": {
        "name": "régime de responsabilité",
        "definition": (
            "Allocation de la responsabilité JURIDIQUE d'un dommage lié à un système, une "
            "activité ou une infrastructure réglementée : règle qui attribue ou modifie qui "
            "répond d'un dommage, établit une obligation de réparation, aménage ou renverse le "
            "fardeau de la preuve en matière de responsabilité, ou impose une assurance "
            "obligatoire destinée à garantir cette responsabilité. N'inclut PAS les amendes, "
            "sanctions pénales ou administratives, confiscations, ni les retraits ou "
            "suspensions de permis en cas d'infraction : ce sont des mécanismes d'exécution / "
            "sanction, pas une allocation de responsabilité civile. N'inclut pas non plus le "
            "simple renvoi implicite aux règles générales du droit privé (CO, CC) sans régime "
            "spécifique créé par la loi elle-même."
        ),
        "include": [
            "régime de responsabilité civile spécifique, créé ou modifié par la loi, pour une "
            "activité, un produit ou un système donné (ex. responsabilité de l'exploitant, du "
            "fabricant, du fournisseur) ;",
            "obligation légale de réparer un dommage causé par un système ou une activité "
            "réglementée, y compris un régime de responsabilité causale (sans faute) instauré "
            "par la loi ;",
            "aménagement ou renversement du fardeau de la preuve en matière de responsabilité ;",
            "assurance obligatoire destinée à garantir la réparation d'un dommage lié à "
            "l'activité réglementée.",
        ],
        "exclude": [
            "les amendes, sanctions pénales ou administratives, confiscations, retraits ou "
            "suspensions de permis en cas d'infraction sont des mécanismes d'exécution / "
            "sanction, qui ne relèvent d'aucun des 7 instruments de ce portfolio — ce n'est PAS "
            "un régime de responsabilité au sens de cette catégorie ;",
            "un simple renvoi implicite aux règles générales de la responsabilité civile (CO, "
            "CC), sans régime spécifique créé par la loi elle-même pour l'activité réglementée, "
            "n'est pas à coder ici ;",
            "l'obligation ou l'interdiction dont la violation est éventuellement sanctionnée "
            "doit être codée séparément sous OBLIGATION ou PROHIBITION_BAN, pas ici.",
        ],
        "examples_oui": [
            (
                "L'exploitant du système répond, indépendamment de toute faute, du dommage "
                "causé par un dysfonctionnement de celui-ci.",
                "régime de responsabilité causale (sans faute) créé spécifiquement par la loi "
                "pour cette activité.",
            ),
            (
                "Les exploitants doivent souscrire une assurance couvrant les dommages causés "
                "par le système avant sa mise en service.",
                "assurance obligatoire garantissant la responsabilité pour l'activité "
                "réglementée.",
            ),
        ],
        "examples_non": [
            (
                "Quiconque contrevient intentionnellement aux dispositions de l'art. 15 est "
                "puni d'une amende de 100 000 francs au plus.",
                "sanction pénale/administrative (mécanisme d'exécution), pas une allocation de "
                "responsabilité pour un dommage.",
            ),
            (
                "L'autorité compétente peut retirer l'autorisation d'exploitation en cas de "
                "violation grave et répétée des exigences de sécurité.",
                "mesure d'exécution (retrait de permis), pas un régime de responsabilité.",
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
        f"Réponds uniquement à la question suivante : le texte de l'article met-il lui-même en "
        f"place le mécanisme « {d['name']} », ou habilite-t-il explicitement une autorité à "
        f"utiliser précisément ce mécanisme ?\n\n"
        "Ne déduis JAMAIS la présence de cet instrument à partir de ce qu'une autorité pourrait "
        "faire dans l'exercice d'une compétence générale. Des formulations comme « peut prendre "
        "les mesures nécessaires », « règle les modalités » ou « peut prévoir des exceptions » "
        "ne suffisent PAS à elles seules, sauf si le mécanisme précis recherché y est "
        "explicitement nommé.\n\n"
        "Un article de loi suisse est souvent composé de plusieurs alinéas. Examine CHAQUE "
        "alinéa séparément : il suffit qu'UN SEUL alinéa relève de cet instrument précis pour "
        "que l'article entier soit classé OUI, même si les autres alinéas relèvent d'autres "
        "instruments ou n'en contiennent aucun.\n\n"
        "## Définition\n\n"
        f"{d['definition']}\n\n"
        "## Inclure\n\n"
        f"Classe l'article OUI lorsque le texte met en place lui-même, ou habilite "
        f"explicitement une autorité à utiliser précisément, un mécanisme correspondant à "
        f"{d['name']}, notamment :\n"
        f"{include_bullets}\n\n"
        "Les formulations exactes peuvent varier, mais le mécanisme juridique doit être "
        "clairement identifiable dans le texte. Une similarité générale de fonction ou "
        "d'objectif ne suffit PAS.\n\n"
        "## Exclure\n\n"
        "Ne classe PAS l'article ici lorsque le mécanisme qu'il contient relève en réalité d'un "
        "AUTRE instrument de politique publique, n'est pas un instrument de ce portfolio, ou "
        "n'est qu'une compétence générale non exercée dans le texte. En particulier :\n"
        f"{exclude_bullets}\n\n"
        "## Exemples\n\n"
        f"{examples_block}\n\n"
        "## Règle de décision\n\n"
        f"Applique le test suivant : le texte de l'article met-il lui-même en place, ou "
        f"habilite-t-il explicitement une autorité à utiliser précisément, un mécanisme "
        f"correspondant à {d['name']}, tel que défini ci-dessus ?\n"
        "- Si l'article contient plusieurs alinéas ou dispositions, classe-le OUI dès qu'AU "
        "MOINS UN d'entre eux met en place un tel mécanisme. Ne fais JAMAIS la moyenne ni le "
        "vote majoritaire entre alinéas.\n"
        "- Un même article peut parfaitement contenir plusieurs instruments distincts. Pour "
        "cette question, ne réponds OUI que si tu identifies dans le texte un mécanisme "
        "distinct correspondant précisément à l'instrument recherché ici — indépendamment du "
        "fait que d'autres mécanismes, relevant d'autres instruments, soient aussi présents "
        "dans le même article.\n"
        "- Si le mécanisme identifié correspond mieux à un AUTRE instrument (voir section "
        "\"Exclure\" ci-dessus), réponds NON pour CET instrument-ci.\n"
        "- N'infère pas la présence de cet instrument à partir de l'objectif général de la loi, "
        "du titre de l'article, du secteur de politique publique, ou de mécanismes qui "
        "pourraient exister ailleurs dans la législation.\n"
        "- Fonde la décision uniquement sur le contenu de l'article fourni.\n"
        "- En cas d'incertitude, privilégie NON : ne réponds OUI que si le mécanisme "
        "correspondant précisément à cet instrument est explicitement identifiable dans "
        "l'article lui-même.\n\n"
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
