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
        "Mécanisme FORMALISÉ de régulation non contraignante par lequel une autorité publique "
        "cherche à orienter le comportement d'acteurs en les invitant à adopter, respecter ou "
        "suivre volontairement une norme, un engagement, un accord ou un cadre de conduite "
        "déterminé. Le caractère distinctif de cet instrument est que l'ADHÉSION ou la "
        "CONFORMITÉ VOLONTAIRE constitue elle-même le mécanisme de politique publique. "
        "Un instrument volontaire ne désigne donc PAS simplement une activité facultative, "
        "permise ou autorisée. Le fait qu'un acteur soit libre d'exercer une activité, de "
        "participer à un dispositif, de demander une autorisation ou qu'une autorité puisse "
        "autoriser quelque chose ne constitue PAS un instrument volontaire."
    ),
    "include": [
        "codes de conduite élaborés, reconnus ou soutenus par une autorité, auxquels des "
        "acteurs sont invités à adhérer volontairement ;",
        "chartes ou engagements volontaires formalisés par lesquels des acteurs s'engagent "
        "librement à respecter certaines pratiques ou certains standards ;",
        "accords volontaires conclus entre une autorité et des acteurs privés ou publics "
        "(par exemple des engagements sectoriels négociés) ;",
        "standards non contraignants que les acteurs sont explicitement encouragés à adopter "
        "ou respecter volontairement ;",
        "labels ou certifications réellement facultatifs lorsqu'ils sont utilisés comme "
        "mécanisme de politique publique pour encourager l'adoption volontaire de pratiques "
        "ou de standards déterminés.",
    ],
    "exclude": [
        "IMPORTANT : le simple fait qu'une activité soit FACULTATIVE, PERMISE, AUTORISÉE ou "
        "laissée au choix d'un acteur ne constitue PAS un instrument volontaire. "
        "VOLUNTARY signifie que l'adhésion volontaire à une norme, un engagement ou un cadre "
        "de conduite est elle-même utilisée comme mécanisme pour orienter les comportements ;",
        "une disposition permettant ou autorisant une activité, un projet, un essai, une "
        "expérimentation ou un projet pilote N'EST PAS un instrument volontaire, même si les "
        "acteurs sont libres de participer ;",
        "un régime d'autorisation, de permis ou d'admission N'EST PAS un instrument volontaire : "
        "le fait qu'un acteur choisisse librement de demander une autorisation ne transforme "
        "pas le régime juridique en mécanisme volontaire ;",
        "un bac à sable réglementaire, un régime expérimental ou des essais encadrés ne "
        "constituent PAS un instrument volontaire du seul fait que la participation est "
        "facultative ; ils relèvent de PLANNING_EVALUATION lorsque les critères correspondants "
        "sont remplis ;",
        "une disposition indiquant qu'une autorité « peut » prendre une décision, accorder une "
        "autorisation, prévoir une dérogation ou exercer une compétence N'EST PAS un instrument "
        "volontaire : « peut » exprime ici un pouvoir ou une marge d'appréciation de "
        "l'autorité, pas une adhésion volontaire à un mécanisme de régulation ;",
        "une dérogation ou une exemption à une règle contraignante N'EST PAS un instrument "
        "volontaire, même lorsqu'un acteur peut choisir d'en bénéficier ;",
        "la possibilité pour un acteur d'exercer un droit, de déposer une demande, de recourir "
        "à une procédure ou d'utiliser un service N'EST PAS un instrument volontaire ;",
        "une simple campagne d'information, de sensibilisation, de conseil ou "
        "d'accompagnement, sans mécanisme formalisé d'adhésion ou de conformité volontaire, "
        "N'EST PAS un instrument volontaire de ce portfolio ;",
        "toute mesure qui impose une obligation juridiquement contraignante à un acteur relève "
        "de OBLIGATION, pas de VOLUNTARY ;",
        "tout mécanisme financier (subvention, aide, taxe, exonération) relève de "
        "TAXES_SUBSIDIES, même lorsque la participation au programme ou la demande d'aide est "
        "facultative ;",
        "toute interdiction relève de PROHIBITION_BAN ;",
        "la simple création ou organisation d'une autorité, sans mécanisme formalisé "
        "d'adhésion ou de conformité volontaire, n'est pas un instrument.",
    ],
    "examples_oui": [
        (
            "Les fournisseurs peuvent adhérer à un code de conduite élaboré en collaboration "
            "avec l'autorité compétente et s'engager volontairement à respecter les standards "
            "qu'il contient.",
            "code de conduite formalisé dont l'adhésion et le respect volontaires constituent "
            "le mécanisme utilisé pour orienter le comportement des acteurs.",
        ),
        (
            "Les entreprises du secteur peuvent conclure avec la Confédération des accords "
            "volontaires par lesquels elles s'engagent à réduire leur consommation "
            "énergétique selon des objectifs définis en commun.",
            "accord volontaire formalisé : les acteurs adoptent volontairement des engagements "
            "destinés à orienter leur comportement.",
        ),
        (
            "L'autorité peut établir un label auquel les fournisseurs peuvent adhérer "
            "volontairement lorsqu'ils respectent le standard non contraignant défini par "
            "le programme.",
            "label facultatif utilisé comme mécanisme formalisé pour encourager l'adoption "
            "volontaire d'un standard.",
        ),
    ],
    "examples_non": [
        (
            "L'autorité peut autoriser des essais de durée limitée avec des véhicules équipés "
            "d'un système d'automatisation.",
            "autorisation d'une expérimentation : le fait que l'essai soit facultatif ou "
            "autorisé ne constitue pas une adhésion volontaire à une norme ou à un engagement.",
        ),
        (
            "L'autorité peut prévoir des dérogations aux dispositions applicables dans le "
            "cadre d'un projet pilote.",
            "pouvoir discrétionnaire et régime de dérogation, pas mécanisme de régulation par "
            "adhésion volontaire.",
        ),
        (
            "Les exploitants peuvent demander une autorisation pour utiliser le système dans "
            "les conditions prévues par la présente loi.",
            "le dépôt facultatif d'une demande et l'existence d'un régime d'autorisation ne "
            "constituent pas un instrument volontaire.",
        ),
        (
            "La Confédération peut soutenir des campagnes d'information visant à sensibiliser "
            "le public aux risques liés à une nouvelle technologie.",
            "simple information ou sensibilisation, sans norme, engagement ou cadre de "
            "conduite auquel des acteurs sont invités à adhérer volontairement.",
        ),
        (
            "Les exploitants doivent informer les autorités de tout incident grave dans un "
            "délai de 24 heures.",
            "obligation juridiquement contraignante de notification (OBLIGATION), pas un "
            "mécanisme de conformité volontaire.",
        ),
        (
            "La présente loi a pour but de promouvoir l'innovation.",
            "énoncé de but général sans mécanisme concret de régulation volontaire.",
        ),
    ],
},
    "TAXES_SUBSIDIES": {
        "name": "taxes et subventions",
        "definition": (
    "Mécanisme FINANCIER par lequel une autorité publique impose un prélèvement monétaire "
    "(taxe ou impôt) ou accorde un avantage financier (subvention, aide financière, "
    "exonération ou avantage fiscal) afin d'orienter ou de soutenir un comportement ou une "
    "activité. Pour être classé OUI, l'article doit contenir un TRANSFERT, PRÉLÈVEMENT ou "
    "AVANTAGE MONÉTAIRE identifiable. Une mesure qui encourage ou décourage un comportement "
    "sans mécanisme financier n'est PAS une taxe ou une subvention."
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
            "une interdiction, une obligation ou toute autre règle contraignante n'est PAS une "
"taxe ou une subvention simplement parce qu'elle crée une incitation ou une "
"désincitation comportementale ;",

"les expressions telles que « but lucratif », « valeur commerciale », « activité "
"économique », « commercialisation », « prix » ou « coût » ne constituent PAS en "
"elles-mêmes un mécanisme financier public ;",

"si aucun prélèvement, versement, avantage fiscal ou transfert monétaire par une "
"autorité publique ne peut être identifié dans le texte, répondre NON ;",
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
    "Mécanisme formalisé dont l'objet DIRECT est de planifier, tester, surveiller ou "
    "évaluer une politique, une activité, un système ou un risque. Il doit exister dans "
    "l'article un exercice identifiable de planification, d'expérimentation, d'audit, "
    "d'analyse d'impact, de monitoring systématique ou d'évaluation. Le simple fait qu'une "
    "règle définisse, encadre ou contrôle un comportement ne constitue PAS une planification "
    "ou une évaluation."
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
            "une interdiction ou une obligation n'est PAS un instrument de planification ou "
"d'évaluation simplement parce qu'elle vise à prévenir, contrôler ou réduire un risque ;",

"une définition, une liste de critères, une exception ou la délimitation précise d'une "
"règle ne constitue PAS une évaluation ;",

"si aucun plan, audit, essai encadré, analyse d'impact, monitoring systématique ou "
"évaluation identifiable n'est prévu par l'article, répondre NON ;",
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
            "OBLIGATION désigne EXCLUSIVEMENT une obligation POSITIVE : un acteur est juridiquement "
    "tenu de FAIRE, FOURNIR, METTRE EN PLACE, MAINTENIR, GARANTIR ou RESPECTER une exigence "
    "positive déterminée. Une obligation de NE PAS FAIRE quelque chose est, dans cette "
    "taxonomie, une PROHIBITION_BAN et doit toujours être classée NON ici."
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
            "une tâche, compétence ou obligation imposée à une autorité uniquement pour organiser "
"son activité administrative (consulter une autre autorité, obtenir son accord, transmettre "
"un dossier, coordonner une procédure, prendre une décision) n'est PAS une OBLIGATION au "
"sens de cet instrument. Une autorité publique peut toutefois être destinataire d'une "
"OBLIGATION lorsqu'elle est elle-même soumise à une exigence substantielle de politique "
"publique ;",
            "une formulation « il est interdit de », « ne peut pas », « nul ne peut » ou toute "
"autre obligation de s'abstenir relève exclusivement de PROHIBITION_BAN ;",

"ne transforme JAMAIS une interdiction en obligation au motif qu'elle impose juridiquement "
"à l'acteur de respecter l'interdiction ;",
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
    "Règle qui attribue explicitement la RESPONSABILITÉ JURIDIQUE pour un dommage ou un "
    "préjudice et détermine qui doit en répondre, le réparer ou l'indemniser. L'article doit "
    "lui-même établir ou modifier un mécanisme de responsabilité. L'existence d'une "
    "obligation ou d'une interdiction dont la violation POURRAIT entraîner une responsabilité "
    "ailleurs dans le droit ne suffit jamais."
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
            "une interdiction ou une obligation n'établit PAS un régime de responsabilité simplement "
"parce que sa violation pourrait entraîner des conséquences juridiques ;",

"une exception à une interdiction ne constitue PAS une exonération de responsabilité au "
"sens de cet instrument ;",

"ne déduis JAMAIS une responsabilité causale, pour faute ou sans faute si l'article ne "
"l'établit pas explicitement ;",

"si l'article n'identifie aucun dommage, aucune obligation de réparation ou d'indemnisation, "
"et aucune attribution explicite de responsabilité juridique, répondre NON ;",
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
        f"Réponds uniquement à la question suivante : cet article contient-il l'instrument "
        f"« {d['name']} » ?\n\n"

        "Un article de loi suisse est souvent composé de plusieurs alinéas. Examine CHAQUE "
        "alinéa séparément : il suffit qu'UN SEUL alinéa relève de cet instrument précis pour "
        "que l'article entier soit classé OUI, même si les autres alinéas relèvent d'autres "
        "instruments ou n'en contiennent aucun.\n\n"

        "## Définition\n\n"
        f"{d['definition']}\n\n"

        "## Inclure\n\n"
        f"Classe l'article OUI lorsqu'il contient lui-même au moins un mécanisme correspondant "
        f"précisément à {d['name']}, notamment :\n"
        f"{include_bullets}\n\n"
        "Les formulations juridiques exactes peuvent varier. Réponds OUI uniquement si le "
        "texte contient clairement le mécanisme défini ci-dessus. Une similarité générale "
        "de fonction ou d'objectif ne suffit pas.\n\n"

        "## Exclure\n\n"
        "Ne classe PAS l'article ici lorsque le mécanisme qu'il contient relève en réalité "
        "d'un AUTRE instrument de politique publique, ou d'aucun instrument. En particulier :\n"
        f"{exclude_bullets}\n\n"

        "## Délégation du pouvoir réglementaire\n\n"
        "Une loi peut déléguer au Conseil fédéral ou à une autre autorité le soin de préciser "
        "ultérieurement une réglementation. Une telle délégation compte comme l'instrument "
        "recherché UNIQUEMENT lorsque la loi identifie déjà clairement la NATURE de cet "
        "instrument et délègue seulement sa précision ou sa mise en œuvre.\n\n"
        "Par exemple, « le Conseil fédéral fixe les exigences auxquelles les exploitants "
        "doivent satisfaire » établit déjà un régime d'exigences contraignantes : "
        "OBLIGATION peut être OUI. De même, une disposition autorisant explicitement le "
        "Conseil fédéral à interdire une pratique peut constituer PROHIBITION_BAN.\n\n"
        "En revanche, une habilitation générale telle que « le Conseil fédéral règle les "
        "modalités », « édicte les dispositions d'exécution » ou « peut prendre les mesures "
        "nécessaires » ne permet PAS d'inférer un instrument précis.\n\n"
        "RÈGLE : si la loi détermine clairement l'instrument mais délègue uniquement ses détails, "
"considère que cet instrument est présent. Si le choix même de l'instrument est laissé à "
"l'autorité, réponds NON.\n\n"

        "## Ne pas raisonner par effet ou implication\n\n"
        "Identifie uniquement le mécanisme juridique contenu dans l'article. Ne réponds PAS "
        "OUI simplement parce que la disposition produit un effet similaire à l'instrument "
        "recherché, encourage ou décourage un comportement, pourrait entraîner une sanction "
        "ou une responsabilité, ou contribue à un objectif auquel cet instrument pourrait "
        "également contribuer.\n\n"
        "Une conséquence possible, un effet comportemental ou une implication juridique "
        "indirecte ne suffit JAMAIS. Pour répondre OUI, un mécanisme correspondant précisément "
        "à la définition de l'instrument recherché doit pouvoir être identifié dans le texte.\n\n"

        "## Exemples\n\n"
        f"{examples_block}\n\n"

        "## Règle de décision\n\n"
        f"Applique le test suivant : l'article contient-il lui-même, ou délègue-t-il "
        f"explicitement, un mécanisme correspondant précisément à {d['name']}, tel que défini "
        "ci-dessus ?\n"
        "- Si l'article contient plusieurs alinéas ou dispositions, classe-le OUI dès qu'AU "
        "MOINS UN d'entre eux contient un tel mécanisme. Ne fais JAMAIS la moyenne ni le vote "
        "majoritaire entre alinéas.\n"
        "- Un même article peut contenir plusieurs instruments distincts. Pour ce prompt, "
        "réponds OUI uniquement si tu peux identifier un mécanisme correspondant précisément "
        "à l'instrument recherché.\n"
        "- Ne déduis jamais cet instrument de ce qu'une autorité POURRAIT faire sur la base "
        "d'une compétence générale. Une délégation ne compte que si le type d'instrument est "
        "déjà identifiable dans le texte.\n"
        "- N'infère pas la présence de cet instrument à partir de l'objectif général de la loi, "
        "du titre de l'article, du secteur de politique publique ou d'un simple effet indirect.\n"
        "- Fonde la décision uniquement sur le contenu de l'article fourni.\n"
        "- En cas d'incertitude, privilégie NON, sauf si un mécanisme correspondant précisément "
        "à cet instrument peut être identifié dans l'article lui-même.\n\n"

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
