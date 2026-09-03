# src/run_target_prompts.py
# Un prompt système dédié par cible (target), sur le même modèle que
# run_inst_prompts.py : une question binaire fermée par cible, au lieu d'une
# seule question générique couvrant les 10 cibles à la fois.
#
# Taxonomie des 10 cibles opérationnelles : PoC PDF (Kaiser, Hinterleitner,
# Tamò-Larrieux & Caprettini), section 4.2.2, grille à 4 quadrants
# Enabling/Safeguarding x Upstream/Downstream. Cette version remplace la
# précédente répartition à 10 cibles de ce même module sur deux points :
#   - les anciennes cibles "Données & Vie privée" et "Propriété
#     intellectuelle & Droits créatifs" (volet input) fusionnent en une
#     seule cible Safeguarding x Upstream, DATA_PRIVACY_IP, qui couvre tout
#     droit ou toute protection attaché aux données ou contenus utilisés en
#     AMONT pour construire ou faire fonctionner un système d'IA — y compris
#     désormais les règles de propriété intellectuelle sur les données
#     d'entraînement, qui n'appartiennent plus à la cible d'accès aux
#     données ;
#   - l'ancienne cible unique "Risques à hauts enjeux" se scinde en deux
#     cibles Safeguarding x Downstream distinctes : OUTPUT_HARMS (un
#     dommage causé par un résultat concret — décision, contenu, action —
#     produit par un système d'IA) et SOCIETAL_HARMS (un effet agrégé,
#     collectif ou systémique de l'usage généralisé de l'IA) ;
#   - la cible ACCOUNTABILITY_TRANSPARENCY s'élargit : elle couvre
#     désormais aussi la supervision humaine, la traçabilité technique et le
#     droit de recours, qu'elle excluait auparavant — tout mécanisme qui
#     rend le fonctionnement ou la décision d'un système scrutable, plutôt
#     que la seule information de la personne concernée ou la seule
#     documentation technique publique.
# La taxonomie de run5_prompts.TARGET_CODES est une version antérieure,
# distincte de celle-ci — ne pas la réutiliser pour ce pipeline.
#
# Cadrage des prompts : éviter les correctifs lexicaux propres à un domaine
# (ex. "véhicule automobile" != "automatisé"). Le corpus couvre des domaines
# beaucoup trop variés (circulation routière, analyses génétiques, marchés
# publics...) pour qu'un correctif pensé pour un domaine particulier
# généralise aux autres — au mieux inutile, au pire un nouveau piège. La
# structure retenue est donc :
#   1. Un bloc de CONTEXTE partagé, identique pour les 10 cibles, qui
#      explique en détail et avec un vocabulaire riche ce qu'est
#      l'intelligence artificielle et selon quelle logique les États la
#      régulent (Enabling : promouvoir son développement ; Safeguarding :
#      encadrer les risques qu'elle pose). Ce bloc ancre le lexique de l'IA
#      sans jamais s'appuyer sur un domaine d'application particulier.
#   2. Un bloc CIBLE, propre à chaque code, avec une définition précise et
#      des exemples concrets ILLUSTRANT des mesures qui satisferaient
#      réellement cette cible (pas des pièges négatifs d'un domaine tiers).
#      Le titre de ce bloc utilise un intitulé délibérément plus spécifique
#      à l'IA que le nom officiel de la cible dans le document (ex. "Accès
#      aux données & ressources pour le développement de l'IA" plutôt que
#      "Accès aux données & Ressources") : un LLM ne voit qu'un seul prompt
#      à la fois et n'a aucune connaissance des 9 autres cibles, donc plus
#      l'intitulé qu'on lui montre est explicitement ancré dans l'IA, moins
#      il risque de dériver vers une lecture générique du problème public.
#      Le nom OFFICIEL de la cible (celui du document, utilisé pour les
#      colonnes de résultat via TARGET_CODES) reste distinct et inchangé.
#   3. Chaque bloc CIBLE est AUTOSUFFISANT : ses exclusions ("ne satisfont
#      pas cette cible...") décrivent directement le contenu du cas exclu
#      (ex. "une exigence qui protège l'intégrité technique du système
#      contre une attaque, sans lien avec le résultat qu'il produit") au
#      lieu de renvoyer au nom d'une autre cible (ex. "cela relève de la
#      cible Sécurité & Robustesse, pas de celle-ci"). Le modèle ne recevant
#      qu'un seul prompt cible à la fois, une exclusion qui ne fait sens que
#      par comparaison avec une cible qu'il ne voit pas est inutilisable —
#      voire trompeuse, puisqu'elle laisse croire à tort que le cas exclu
#      est pertinent ailleurs sans jamais dire pourquoi il ne l'est pas ici.
#   4. Un ANCRAGE par défaut sceptique : le LLM part de la position "à
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
# Bloc de contexte partagé par les 10 cibles : ce qu'est l'IA, et la logique
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
    la vie privée ou à la propriété intellectuelle sur les données qui
    servent à l'entraîner, défaillances techniques face à une attaque ou
    une panne, manque de traçabilité ou de contestabilité de son
    fonctionnement, dommages causés par ses décisions ou ses contenus, ou
    préjudices sociétaux plus larges liés à sa diffusion à grande échelle.
""")

# ---------------------------------------------------------------------------
# Les 10 cibles opérationnelles, dans l'ordre des 4 quadrants du PoC PDF
# (Enabling x Upstream, Safeguarding x Upstream, Enabling x Downstream,
# Safeguarding x Downstream).
#
# `name`        : intitulé OFFICIEL de la cible tel que nommé dans le
#                 document — c'est celui-ci qui doit apparaître dans les
#                 résultats (TARGET_CODES, colonnes de sortie, etc.).
# `prompt_label`: intitulé montré au LLM en tête du bloc cible, délibérément
#                 plus spécifique à l'IA que `name` (voir note en tête de
#                 fichier). N'est utilisé que dans le prompt, jamais dans
#                 les résultats.
# `definition`  : ce que doit faire la norme pour satisfaire cette cible,
#                 suivi de ses exclusions — décrites de façon autosuffisante,
#                 sans jamais nommer une autre cible.
# `examples`    : exemples concrets ILLUSTRANT des mesures qui satisferaient
#                 réellement cette cible (pas des contre-exemples).
# ---------------------------------------------------------------------------

TARGET_DEFINITIONS: "OrderedDict[str, dict]" = OrderedDict(
    [
        (
            "RESEARCH_INNOVATION",
            {
                "name": "Recherche & Innovation",
                "prompt_label": "Recherche & innovation en intelligence artificielle",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme prévoit une
                    mesure par laquelle l'État promeut ou soutient
                    activement la recherche en intelligence artificielle :
                    financement de projets de recherche, création ou
                    financement d'un programme ou d'une infrastructure de
                    recherche publique, subvention à la recherche versée à
                    des entreprises (y compris des start-up) ou à des
                    institutions, ou mise en place d'un programme réunissant
                    chercheurs, développeurs et société civile autour de la
                    recherche en intelligence artificielle. Le mot «
                    recherche » se comprend au sens strict : produire ou
                    faire progresser la connaissance scientifique ou
                    technique sur l'intelligence artificielle. Une mesure
                    qui favorise seulement l'usage, l'adoption,
                    l'expérimentation en conditions réelles ou l'entrée sur
                    le marché de l'intelligence artificielle — sans financer
                    ni organiser une activité de recherche — ne satisfait
                    pas cette cible : ce sont d'autres mesures, pour
                    d'autres cibles, pas celle-ci.
                """) + " " + _p("""
                    Ne satisfont donc PAS cette cible, même quand leur objet
                    touche à l'intelligence artificielle, à un système
                    automatisé, à des données ou à une technologie : un
                    régime d'autorisation d'essais, un bac à sable
                    réglementaire ou une phase pilote destinés à faciliter
                    l'expérimentation ou la mise sur le marché, une fonction
                    d'évaluation ou de conseil sur les risques et
                    opportunités de nouvelles technologies, une obligation
                    de sécurité, de traçabilité ou d'enregistrement imposée
                    à un système déjà autorisé ou déployé (boîte noire,
                    journal d'événements), une obligation de notification ou
                    d'analyse d'incident, les règles générales déterminant
                    la base légale requise pour traiter des données
                    personnelles, une procédure de contrôle de sécurité de
                    personnes, l'attribution d'un droit, une définition
                    juridique, l'institution ou l'organisation d'une
                    autorité (nomination, statut, budget, organe de
                    surveillance, de médiation ou de traitement de
                    plaintes), ou une exception au droit d'auteur (usage
                    privé, accessibilité aux personnes handicapées, durée de
                    protection, mesures anti-contournement). Ces
                    dispositions ne financent, n'organisent et ne créent
                    aucune activité de recherche — même si le texte emploie
                    incidemment des mots comme « innovation », « recherche
                    », « technologie » ou « nouvelles technologies ». En cas
                    de doute, demande-toi : cet article finance-t-il,
                    organise-t-il ou crée-t-il concrètement une activité de
                    recherche en intelligence artificielle (OUI), ou
                    s'agit-il d'autre chose — adoption, expérimentation,
                    mise sur le marché, sécurité, gouvernance, surveillance
                    (NON) ? Un rapprochement lexical avec le vocabulaire de
                    cette cible ne suffit jamais : ta justification doit
                    citer l'élément précis du texte qui finance, organise ou
                    crée une activité de recherche — jamais reformuler la
                    définition de la cible elle-même. Si tu ne peux pas
                    pointer cet élément précis, la réponse est NON.
                """),
                "examples": [
                    "Un fonds public finançant des projets de recherche en intelligence artificielle dans les universités.",
                    "La création d'un centre national de recherche dédié à l'intelligence artificielle.",
                    "Une subvention de l'État à des start-up ou des PME pour financer leurs propres activités de recherche en intelligence artificielle.",
                    "Un programme public organisant des rencontres entre chercheurs, développeurs et société civile pour faire avancer la recherche en intelligence artificielle.",
                ],
            },
        ),
        (
            "SKILLS_HUMAN_CAPITAL",
            {
                "name": "Compétences & Capital humain",
                "prompt_label": "Compétences & capital humain pour l'intelligence artificielle",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si l'État, à travers un
                    dispositif d'enseignement ou de formation (école,
                    gymnase, université, haute école spécialisée, formation
                    professionnelle, formation continue, bourse d'études),
                    organise ou finance l'apprentissage de compétences
                    numériques dont le contenu porte spécifiquement sur
                    l'intelligence artificielle, la science des données ou
                    le calcul informatique. Il s'agit d'une mesure par
                    laquelle l'État outille une partie de la population —
                    élèves, étudiants, professionnels, employés publics —
                    de compétences en IA.
                """) + " " + _p("""
                    Cette cible ne porte PAS sur les cas où un texte EXIGE
                    une compétence, une qualification ou une expertise
                    (professionnelle, technique, psychologique) comme
                    condition pour faire, obtenir ou exercer autre chose —
                    par exemple obtenir un permis de conduire, exercer un
                    métier réglementé, obtenir un agrément ou un certificat
                    de sécurité, ou réaliser une analyse. Une exigence de
                    compétence n'est pas une mesure de formation : seul un
                    dispositif où l'État organise, finance ou dispense
                    lui-même un enseignement dont le contenu porte sur l'IA,
                    les données ou le calcul satisfait cette cible. Si le
                    texte ne décrit pas un tel dispositif d'enseignement ou
                    de formation, la réponse est NON.
                """),
                "examples": [
                    "Un programme de formation continue en science des données et en intelligence artificielle pour des employés de l'administration.",
                    "La création d'une filière universitaire spécialisée en apprentissage automatique.",
                    "Une bourse d'études destinée à financer un cursus en intelligence artificielle.",
                    "L'intégration de l'enseignement de la programmation et de l'intelligence artificielle dans les programmes du gymnase ou de l'école obligatoire.",
                    "Un programme de reconversion professionnelle financé par l'État pour former des travailleurs aux métiers de la science des données.",
                ],
            },
        ),
        (
            "DATA_ACCESS_RESOURCES",
            {
                "name": "Accès aux données & Ressources",
                "prompt_label": "Accès aux données & ressources pour le développement de l'IA",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si l'État MET À DISPOSITION,
                    dans les faits, une ressource de données ou une
                    ressource informationnelle destinée à la recherche, à
                    l'entraînement, au développement ou aux tests de
                    systèmes d'intelligence artificielle — un jeu de données
                    ouvertes ou publiques, une plateforme ou un espace de
                    partage de données, un jeu de données annotées, ou un
                    accès facilité à des données de haute qualité pour
                    entraîner ou évaluer un modèle. L'essentiel est que la
                    norme ORGANISE ou FINANCE la mise à disposition
                    pratique d'une telle ressource : elle ne pose ni ne
                    modifie aucune condition juridique, aucun droit et
                    aucune protection sur cette donnée — elle se contente de
                    la rendre accessible dans les faits à ceux qui
                    construisent ou entraînent des systèmes d'IA.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible : une norme qui donne à un
                    organe de l'État — autorité de surveillance, police,
                    service de renseignement, assureur public, régulateur
                    sectoriel — un accès ou un pouvoir de traitement sur des
                    données pour l'accomplissement de SES propres tâches ;
                    dans ce contexte, l'État n'est pas un développeur de
                    systèmes d'IA, et une telle habilitation ne rend aucune
                    donnée plus accessible à ceux qui en construisent, même
                    si le texte emploie des mots comme « traiter des
                    données », « données sensibles » ou « accès aux
                    données ». Ne satisfait pas non plus cette cible une
                    règle qui pose, modifie ou retire une CONDITION
                    JURIDIQUE (base légale, consentement, licence,
                    protection par un droit, exception à ce droit)
                    applicable à des données ou à des contenus utilisés
                    comme intrant pour l'IA — qu'elle en restreigne ou au
                    contraire en facilite la réutilisation : une telle règle
                    régule un droit ou une protection, elle ne se contente
                    pas d'organiser ou de financer la disponibilité
                    pratique d'une ressource, et ne satisfait donc pas cette
                    cible. Ne satisfait pas non plus cette cible une mesure
                    qui finance l'ACHAT ou l'USAGE, par une entreprise ou
                    une administration, d'un système d'intelligence
                    artificielle déjà construit : cette cible concerne la
                    ressource en données mobilisée pour CONSTRUIRE ou
                    ENTRAÎNER un système, pas le financement de son usage
                    final une fois construit. En cas de doute, demande-toi :
                    cet article organise-t-il ou finance-t-il concrètement
                    la mise à disposition pratique d'une ressource de
                    données pour la recherche, l'entraînement ou le test de
                    systèmes d'IA, SANS poser de condition juridique sur son
                    utilisation (OUI), ou s'agit-il d'un pouvoir d'accès
                    accordé à une administration pour ses propres tâches,
                    d'une règle qui pose ou modifie un droit ou une
                    protection sur des données ou contenus utilisés comme
                    intrant, ou d'un financement de l'usage d'un système
                    déjà construit (NON) ? Un rapprochement lexical avec le
                    mot « données » ne suffit jamais : ta justification doit
                    citer l'élément précis du texte qui organise ou finance
                    cette mise à disposition — jamais reformuler la
                    définition de la cible elle-même. Si tu ne peux pas
                    pointer cet élément précis, la réponse est NON.
                """),
                "examples": [
                    "Une obligation faite aux administrations de mettre à disposition des jeux de données ouvertes utilisables pour entraîner des systèmes d'intelligence artificielle.",
                    "Un portail public de données ouvertes (open data) explicitement présenté comme une ressource pour la recherche ou le développement en intelligence artificielle.",
                    "Un cadre légal organisant un espace de partage de données de santé anonymisées à des fins de recherche en intelligence artificielle.",
                    "Un programme finançant la constitution ou l'annotation d'un jeu de données destiné à l'entraînement ou à l'évaluation de modèles d'intelligence artificielle.",
                    "Une plateforme publique donnant à des chercheurs ou des start-up un accès facilité à des données de haute qualité pour entraîner leurs modèles.",
                ],
            },
        ),
        (
            "COMPUTE_INFRASTRUCTURE",
            {
                "name": "Calcul & Infrastructure",
                "prompt_label": "Calcul & infrastructure pour l'entraînement et le fonctionnement de systèmes d'IA",
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme finance,
                    construit ou facilite concrètement l'accès à la
                    puissance de calcul ou aux ressources matérielles
                    nécessaires pour entraîner (training) ou faire
                    fonctionner (inference) des systèmes d'intelligence
                    artificielle. Concrètement, cela recouvre : des puces
                    spécialisées pour le calcul intensif (GPU, TPU,
                    accélérateurs IA) et leur approvisionnement ; des
                    serveurs, clusters ou supercalculateurs de calcul haute
                    performance (high-performance computing) ; des capacités
                    de cloud computing (crédits cloud, hébergement) réservées
                    ou dédiées à l'entraînement ou au déploiement de
                    modèles ; des centres de données dont l'usage visé est
                    le calcul pour l'IA ; une stratégie de « cloud
                    souverain » ou de souveraineté numérique en matière de
                    calcul destinée aux acteurs de l'IA ; ou un
                    approvisionnement énergétique prioritaire ou dédié à des
                    installations de calcul pour l'IA. Une mesure satisfait
                    cette cible dès lors qu'elle finance l'achat ou la
                    construction de cette capacité de calcul, en subventionne
                    l'accès pour des chercheurs, start-up ou PME, ou met en
                    place un programme ou une stratégie organisant cet
                    accès.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible les normes qui régulent
                    l'infrastructure, l'énergie, le numérique ou les
                    télécommunications EN GÉNÉRAL, sans lien spécifique avec
                    le calcul pour l'IA — même quand le texte emploie
                    littéralement le mot « infrastructure ». Par exemple :
                    une règle d'aménagement du territoire, de construction
                    ou de protection de l'environnement encadrant
                    l'implantation de bâtiments ou de centres de données,
                    sans référence au calcul pour l'IA ; une politique
                    énergétique générale (tarifs de l'électricité,
                    transition énergétique, réseau électrique) sans lien
                    avec des installations de calcul pour l'IA ; le
                    déploiement de réseaux de télécommunication (fibre
                    optique, 5G, téléphonie) motivé par la connectivité
                    générale de la population ; une définition ou un
                    inventaire des « infrastructures critiques » (énergie,
                    eau, transport, télécoms) dont l'objet est de PROTÉGER
                    ces infrastructures contre une attaque ou une
                    défaillance, et non de financer ou de faciliter l'accès
                    à une capacité de calcul — une telle protection ne
                    satisfait pas cette cible ; ou une règle générale de
                    gouvernance ou de sécurité des infrastructures
                    numériques de l'État (systèmes d'information,
                    cybersécurité administrative) sans rapport avec la
                    fourniture de capacité de calcul pour l'IA. Le simple
                    emploi des mots « infrastructure », « numérique »,
                    « centre de données », « énergie » ou « calcul » ne
                    suffit jamais : il faut que le texte relie explicitement
                    cette ressource au développement ou au fonctionnement de
                    systèmes d'intelligence artificielle. En cas de doute,
                    demande-toi : cet article finance-t-il, construit-il ou
                    facilite-t-il concrètement l'accès à une capacité de
                    calcul (puces, cloud, supercalcul, centre de données,
                    énergie dédiée) destinée à l'entraînement ou au
                    fonctionnement de systèmes d'IA (OUI), ou s'agit-il
                    d'infrastructure, d'énergie, de numérique ou de
                    télécommunications en général, ou de leur protection
                    contre une attaque ou une défaillance, sans lien
                    spécifique avec le calcul pour l'IA (NON) ? Un
                    rapprochement lexical avec le mot « infrastructure » ne
                    suffit jamais : ta justification doit citer l'élément
                    précis du texte qui finance ou facilite l'accès à une
                    capacité de calcul pour l'IA — jamais reformuler la
                    définition de la cible elle-même. Si tu ne peux pas
                    pointer cet élément précis, la réponse est NON.
                """),
                "examples": [
                    "Un investissement public dans un centre de calcul destiné à l'entraînement de modèles d'intelligence artificielle.",
                    "Une subvention pour l'achat de puces spécialisées (GPU, TPU) par des PME ou des laboratoires de recherche pour développer des systèmes d'intelligence artificielle.",
                    "Une stratégie nationale de « cloud souverain » garantissant aux acteurs nationaux de l'IA l'accès à des capacités de calcul haute performance.",
                    "Un programme mettant à disposition des crédits de calcul cloud (cloud credits) pour des start-up développant des systèmes d'intelligence artificielle.",
                    "Une garantie d'approvisionnement énergétique prioritaire pour des centres de données dédiés à l'entraînement de grands modèles d'intelligence artificielle.",
                ],
            },
        ),
        (
            "DATA_PRIVACY_IP",
            {
                "name": "Données, Vie privée & Propriété intellectuelle",
                "prompt_label": "Protection des données, de la vie privée et de la propriété intellectuelle sur les intrants de l'IA",
                "quadrant": "Safeguarding x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme régule un droit,
                    une protection ou une condition juridique attaché aux
                    données, aux contenus protégés ou aux ressources
                    utilisés COMME INTRANT (input) pour développer,
                    entraîner, tester ou faire fonctionner un système
                    d'intelligence artificielle ou un système automatisé de
                    traitement de données. Trois familles de règles
                    satisfont cette cible. Premièrement, un régime GÉNÉRAL
                    de protection des données personnelles — base légale
                    requise, finalité, proportionnalité, durée de
                    conservation, droit d'accès ou de rectification,
                    obligation de sécurité des données, ou détermination de
                    quels traitements sont soumis à un tel régime : un tel
                    régime général s'applique par construction à tout
                    traitement par un système d'intelligence artificielle ou
                    par un système automatisé, même si le texte ne mentionne
                    jamais l'IA. Deuxièmement, une règle SPÉCIFIQUE à
                    l'utilisation de l'IA ou d'un traitement automatisé pour
                    collecter, croiser, profiler, ré-identifier ou inférer
                    des informations sur des personnes à partir de données.
                    Troisièmement, un droit de propriété intellectuelle, un
                    droit d'auteur, une exception à ce droit, ou une
                    exigence de consentement, de licence ou de traçabilité
                    de provenance portant sur des DONNÉES OU DES ŒUVRES
                    UTILISÉES POUR ENTRAÎNER ou alimenter un système
                    d'intelligence artificielle — que cette règle restreigne
                    ou au contraire permette leur réutilisation (par exemple
                    une exception au droit d'auteur pour la fouille de
                    textes et de données, ou une obligation de rémunérer les
                    titulaires de droits dont les œuvres ont servi à
                    l'entraînement d'un modèle). Dans les trois cas, la
                    norme doit réguler un droit, une protection ou une
                    condition juridique attaché à des données ou à des
                    contenus utilisés EN AMONT, pour construire ou faire
                    fonctionner le système — pas au contenu que ce système
                    produit lui-même en sortie.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible : une norme qui attribue ou
                    étend à une administration, une autorité ou tout autre
                    organe de l'État un DROIT ou un POUVOIR d'accéder à des
                    données, de les collecter ou de les traiter pour
                    l'accomplissement de SES propres tâches (par exemple un
                    accès de la police, du fisc ou d'une assurance sociale à
                    un registre ou une base de données) — une telle
                    habilitation confère un pouvoir, elle ne protège ni
                    n'encadre le traitement au sens de cette cible, même si
                    elle porte sur des données personnelles ou sensibles ;
                    une norme qui régule la GÉNÉRATION de données par un
                    processus qui n'a rien à voir avec l'IA ou un traitement
                    automatisé — par exemple une analyse génétique, un
                    examen médical, un prélèvement biométrique ou une
                    collecte de données sur papier — même lorsque les
                    données ainsi générées sont ensuite conservées ou
                    traitées ; une norme qui se contente d'ORGANISER ou de
                    FINANCER la mise à disposition pratique d'un jeu de
                    données ou d'une ressource informationnelle déjà
                    existante (portail de données ouvertes, plateforme de
                    partage), sans poser aucune condition, protection ou
                    droit sur son utilisation — une telle norme rend une
                    ressource disponible dans les faits, elle ne régule
                    aucun droit ni aucune protection, et ne satisfait donc
                    pas cette cible ; et une règle de propriété
                    intellectuelle ou de droit d'auteur qui porte sur le
                    CONTENU PRODUIT EN SORTIE par un système d'intelligence
                    artificielle — titularité des droits sur une création
                    générée, contrefaçon d'une œuvre préexistante par un
                    contenu généré — une telle règle porte sur le résultat
                    produit par le système, pas sur les données ou œuvres
                    utilisées en amont pour l'entraîner, et ne satisfait
                    donc pas non plus cette cible. Le simple emploi des mots
                    « données », « vie privée », « propriété intellectuelle
                    » ou « droit d'auteur » ne suffit jamais : il faut que
                    le texte régule effectivement un droit, une protection
                    ou une condition juridique attaché à des données ou des
                    contenus utilisés en amont pour construire ou faire
                    fonctionner un système d'IA. En cas de doute,
                    demande-toi : cet article régule-t-il un droit, une
                    protection ou une condition juridique (licéité,
                    consentement, licence, protection par le droit d'auteur,
                    exception à ce droit) attaché à des données ou à des
                    contenus utilisés comme INTRANT pour développer,
                    entraîner ou faire fonctionner un système d'IA (OUI), ou
                    s'agit-il d'un pouvoir d'accès accordé à une
                    administration pour ses propres tâches, d'une génération
                    de données non automatisée, de la simple mise à
                    disposition d'une ressource sans condition juridique
                    attachée, ou d'un droit portant sur le contenu produit
                    EN SORTIE par le système (NON) ? Ta justification doit
                    citer l'élément précis du texte qui régule ce droit ou
                    cette protection — jamais reformuler la définition de la
                    cible elle-même. Si tu ne peux pas pointer cet élément
                    précis, la réponse est NON.
                """),
                "examples": [
                    "Une règle générale fixant les conditions de licéité (base légale, finalité, proportionnalité) de tout traitement de données personnelles, applicable aussi bien à un traitement manuel qu'à un traitement par un système automatisé ou par une intelligence artificielle.",
                    "Une obligation de pseudonymiser ou d'anonymiser des données personnelles avant leur traitement par un système d'intelligence artificielle utilisé à des fins de profilage ou de classification de personnes.",
                    "Une interdiction de ré-identifier une personne à partir de données que l'intelligence artificielle a rendues anonymes.",
                    "Un droit de s'opposer au traitement de ses données personnelles par un système de décision automatisée ou d'intelligence artificielle.",
                    "Une exception au droit d'auteur autorisant la fouille de textes et de données (text and data mining) sur des œuvres déjà licitement accessibles, à des fins d'entraînement de modèles d'intelligence artificielle.",
                    "Une obligation de rémunérer les titulaires de droits d'auteur dont les œuvres ont été utilisées pour entraîner un système d'intelligence artificielle.",
                    "Une exigence de traçabilité de la provenance des données ou des œuvres utilisées pour entraîner un modèle, incluant leur statut au regard du droit d'auteur.",
                ],
            },
        ),
        (
            "SECURITY_ROBUSTNESS",
            {
                "name": "Sécurité & Robustesse",
                "prompt_label": "Sécurité & robustesse technique des systèmes d'intelligence artificielle",
                "quadrant": "Safeguarding x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme impose une
                    exigence dont l'objet spécifique est d'assurer
                    l'INTÉGRITÉ TECHNIQUE d'un système d'intelligence
                    artificielle, de son infrastructure ou de ses données
                    d'entraînement ou de fonctionnement — c'est-à-dire sa
                    sécurité, sa fiabilité, sa résilience ou sa robustesse
                    technique. Cela recouvre deux types de menaces.
                    D'une part, une menace MALVEILLANTE : une intrusion, un
                    accès non autorisé, un piratage, une manipulation
                    malveillante (par exemple un empoisonnement de données
                    ou une attaque adverse visant à tromper un modèle), un
                    sabotage, ou toute autre atteinte à la confidentialité, à
                    l'intégrité ou à la disponibilité du système provoquée
                    par un acteur malveillant. D'autre part, une DÉFAILLANCE
                    non malveillante : une exigence de fiabilité, de
                    résistance à l'erreur, de résilience face à une panne, à
                    des données aberrantes ou à un dysfonctionnement
                    technique, ou un test de performance ou de robustesse
                    destiné à vérifier que le système continue de
                    fonctionner correctement dans des conditions dégradées.
                    Peu importe que le système ou l'infrastructure visé
                    appartienne à l'État ou à un acteur privé : ce qui
                    compte est que la mesure protège l'intégrité technique,
                    la fiabilité ou la robustesse du système lui-même, de
                    son infrastructure ou de ses données — pas les personnes
                    affectées par ce que ce système produit ou décide.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible : une mesure qui protège la
                    confidentialité de données personnelles ou régule
                    généralement leur traitement (base légale, finalité,
                    durée de conservation, droit d'accès ou de
                    rectification) sans viser spécifiquement l'intégrité
                    technique d'un système contre une attaque ou une
                    défaillance — même lorsqu'elle emploie le mot «
                    sécurité » (par exemple une « obligation de sécurité des
                    données ») ; une exigence dont l'objet est de protéger
                    des personnes contre un dommage, une décision incorrecte,
                    discriminatoire ou dangereuse causée par ce qu'un
                    système d'IA produit ou décide — une telle exigence
                    porte sur la conséquence produite par le système pour
                    autrui, pas sur son intégrité technique interne ; une
                    exigence de compréhensibilité, d'explication, de
                    supervision humaine ou de possibilité de contester une
                    décision individuelle, qui organise un contrôle sur la
                    décision et non une protection technique du système ;
                    une exigence de sécurité physique sans lien avec un
                    système d'IA ou automatisé (bâtiment, coffre-fort,
                    protection périmétrique) ; une loi ou une stratégie
                    générale de cybersécurité de l'État ou d'une entreprise
                    sans lien spécifique avec un système d'IA ou un système
                    automatisé de traitement de données ; ou une définition
                    ou un inventaire des infrastructures critiques (énergie,
                    eau, transport, télécoms) dont l'objet est la protection
                    de ces infrastructures en général, sans lien spécifique
                    avec un système d'IA ou automatisé qui les pilote ou les
                    traite. Le simple emploi des mots « sécurité », «
                    robustesse », « résilience », « fiabilité » ou «
                    cybersécurité » ne suffit jamais : il faut que le texte
                    protège spécifiquement l'intégrité technique — contre
                    une attaque ou une défaillance — d'un système d'IA ou
                    d'un système automatisé de traitement de données. En cas
                    de doute, demande-toi : cet article impose-t-il une
                    exigence dont l'objet spécifique est de protéger
                    l'intégrité technique (sécurité, fiabilité, résilience,
                    robustesse) d'un système d'IA ou automatisé contre une
                    attaque malveillante ou une défaillance (OUI), ou
                    s'agit-il de protection des données personnelles en
                    général, de protection des personnes contre les
                    conséquences produites par le système, de supervision
                    humaine d'une décision, ou de sécurité ou de
                    cybersécurité générale sans lien spécifique avec l'IA
                    (NON) ? Un rapprochement lexical avec le vocabulaire de
                    la sécurité ne suffit jamais : ta justification doit
                    citer l'élément précis du texte qui protège l'intégrité
                    technique du système — jamais reformuler la définition
                    de la cible elle-même. Si tu ne peux pas pointer cet
                    élément précis, la réponse est NON.
                """),
                "examples": [
                    "Une obligation de certification de cybersécurité pour les systèmes d'intelligence artificielle utilisés dans une infrastructure critique, destinée à prévenir les intrusions et le piratage.",
                    "Une exigence de test de résistance aux attaques adverses (adversarial attacks) ou à l'empoisonnement de données avant la mise en service d'un système d'intelligence artificielle dans un contexte sensible.",
                    "Une obligation pour les fournisseurs de systèmes d'intelligence artificielle de mettre en place des mesures de protection contre l'accès non autorisé ou le piratage de leurs modèles et infrastructures.",
                    "Une exigence de fiabilité et de résistance aux pannes pour un système d'intelligence artificielle pilotant une infrastructure sensible.",
                    "Une exigence de détection et de notification des incidents de sécurité affectant un système d'intelligence artificielle déployé par une entreprise ou une administration.",
                    "Une exigence de gestion des vulnérabilités techniques d'un système d'intelligence artificielle tout au long de son cycle de vie.",
                ],
            },
        ),
        (
            "AI_DEPLOYMENT",
            {
                "name": "Déploiement de l'IA",
                "prompt_label": "Déploiement, adoption et expérimentation de systèmes d'intelligence artificielle",
                "quadrant": "Enabling x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme encourage,
                    facilite ou impose concrètement l'UTILISATION ou le
                    DÉPLOIEMENT effectif d'un système d'intelligence
                    artificielle déjà disponible, par une entreprise, une
                    administration publique ou une autre organisation.
                    Quatre types de mesures satisfont cette cible.
                    Premièrement, une règle qui pousse ou oblige une
                    administration publique à utiliser directement un
                    système d'intelligence artificielle dans
                    l'accomplissement de ses tâches — par exemple pour
                    traiter des dossiers, rendre un service ou préparer une
                    décision. Deuxièmement, un soutien financier de l'État
                    (subvention, crédit d'impôt, fonds, aide à
                    l'investissement) destiné à aider des entreprises, en
                    particulier des PME, à ACQUÉRIR ou à FINANCER
                    l'utilisation d'un logiciel ou d'un système
                    d'intelligence artificielle dans leur activité — par
                    opposition à un financement de la recherche sur l'IA
                    elle-même. Troisièmement, un assouplissement
                    réglementaire (simplification d'une procédure,
                    suppression d'une autorisation préalable, réduction
                    d'une exigence administrative) dont l'effet est de
                    FACILITER l'utilisation ou le déploiement d'un système
                    d'intelligence artificielle déjà existant par des
                    organisations. Quatrièmement, un dispositif
                    d'expérimentation encadrée — bac à sable réglementaire,
                    autorisation d'essai pilote, autorisation temporaire ou
                    dérogatoire, ou procédure de test avant introduction sur
                    le marché — qui PERMET à une entreprise, une
                    administration ou une autre organisation d'utiliser ou
                    de tester un système d'intelligence artificielle en
                    conditions réelles, même de façon limitée dans le temps,
                    l'espace ou le champ d'application, avant sa
                    généralisation : en autorisant cet usage réel, l'État
                    encourage directement l'adoption et la diffusion du
                    système, tout comme dans les trois cas précédents.
                    L'expérimentation et l'introduction sur le marché sont
                    ainsi traitées comme des étapes du déploiement, pas
                    comme des sujets distincts. Dans les quatre cas, c'est
                    l'UTILISATION du système d'IA par son destinataire final
                    — l'entreprise, l'administration ou l'organisation qui
                    s'en sert, même à titre pilote ou expérimental — qui
                    doit être directement encouragée, financée, autorisée ou
                    facilitée par la norme ; peu importe le secteur (santé,
                    justice, transport, etc.) : ce qui compte est que la
                    mesure vise l'usage réel de l'IA par son destinataire
                    final, pas un autre maillon de la chaîne.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible les mesures qui, bien que
                    liées à l'intelligence artificielle, portent en réalité
                    sur un autre maillon de la chaîne de valeur — même si,
                    par ricochet, elles pourraient elles aussi favoriser
                    l'IA en général. Ne satisfont donc pas cette cible : un
                    financement ou une organisation d'une activité de
                    RECHERCHE en intelligence artificielle, qui ne finance
                    ni n'impose aucun usage concret par une entreprise ou
                    une administration ; un dispositif d'ENSEIGNEMENT ou de
                    FORMATION en intelligence artificielle, qui outille des
                    personnes de compétences sans encourager ni faciliter
                    l'utilisation d'un système par une organisation ; une
                    mesure qui facilite l'accès à des DONNÉES d'entraînement
                    ou à de la PUISSANCE DE CALCUL, qui profite à ceux qui
                    construisent des systèmes d'IA et non à ceux qui les
                    utilisent une fois construits ; ainsi que toute exigence
                    de sécurité, de robustesse, de transparence, de
                    traçabilité, de supervision humaine, de protection des
                    données, de protection de droits fondamentaux ou de
                    protection contre des préjudices, même lorsque cette
                    exigence porte sur un système d'IA déjà utilisé ou
                    déployé, y compris dans le cadre d'un bac à sable ou
                    d'un essai pilote — une telle exigence ENCADRE ou
                    RESTREINT l'usage, elle ne l'encourage ni ne le
                    facilite. Une simple mention de la « transformation
                    numérique », de la « modernisation de l'administration »
                    ou de la « digitalisation », sans référence spécifique à
                    l'intelligence artificielle, ne satisfait pas non plus
                    cette cible. En cas de doute, demande-toi : cet article
                    encourage-t-il, finance-t-il, autorise-t-il ou
                    facilite-t-il concrètement l'UTILISATION ou le
                    DÉPLOIEMENT — y compris à titre pilote ou expérimental —
                    d'un système d'intelligence artificielle par une
                    entreprise, une administration ou une organisation qui
                    s'en sert dans son activité (OUI), ou s'agit-il d'autre
                    chose — recherche, formation, données, calcul, sécurité,
                    gouvernance, protection des droits (NON) ? Un
                    rapprochement lexical avec les mots « adoption », «
                    diffusion », « déploiement », « pilote », « bac à sable
                    » ou « utilisation » ne suffit jamais : ta justification
                    doit citer l'élément précis du texte qui encourage,
                    finance, autorise ou facilite concrètement l'usage d'un
                    système d'IA par son destinataire final — jamais
                    reformuler la définition de la cible elle-même. Si tu ne
                    peux pas pointer cet élément précis, la réponse est NON.
                """),
                "examples": [
                    "Un programme incitant les PME à adopter des outils d'intelligence artificielle dans leur processus de production.",
                    "Une obligation pour les administrations publiques d'intégrer des outils d'intelligence artificielle dans le traitement de certains dossiers.",
                    "Un fonds public octroyant des aides financières aux entreprises pour l'acquisition ou la licence de logiciels d'intelligence artificielle destinés à leur activité.",
                    "Un allègement d'une procédure d'autorisation administrative afin de faciliter le déploiement de systèmes d'intelligence artificielle déjà existants par des organisations.",
                    "Une stratégie nationale fixant un objectif chiffré d'administrations publiques utilisant des systèmes d'intelligence artificielle dans leurs procédures d'ici une date donnée.",
                    "Un bac à sable réglementaire permettant à une entreprise de tester un système d'intelligence artificielle médicale en conditions réelles avant sa mise sur le marché généralisée.",
                    "Une procédure d'autorisation temporaire simplifiée permettant à une administration ou une entreprise de faire un essai pilote d'un dispositif fondé sur l'intelligence artificielle.",
                ],
            },
        ),
        (
            "ACCOUNTABILITY_TRANSPARENCY",
            {
                "name": "Responsabilité & Transparence des systèmes",
                "prompt_label": "Transparence, traçabilité et contestabilité des systèmes d'intelligence artificielle",
                "quadrant": "Safeguarding x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme rend le
                    FONCTIONNEMENT, l'UTILISATION, le PROCESSUS DÉCISIONNEL
                    ou la GOUVERNANCE d'un système d'intelligence
                    artificielle transparent, traçable, explicable,
                    contrôlable, contestable, ou attribuable à un acteur
                    responsable identifié. Six types de mesures satisfont
                    cette cible. Premièrement, une obligation ou un droit
                    d'informer une personne qu'une décision la concernant
                    (refus, octroi, évaluation, classification, sanction) a
                    été prise, en tout ou en partie, par un système
                    d'intelligence artificielle ou par un traitement
                    automatisé de données. Deuxièmement, une obligation
                    d'expliquer le fonctionnement, la logique ou les motifs
                    d'une décision individuelle prise par un tel système.
                    Troisièmement, une obligation de supervision humaine, de
                    validation humaine ou d'intervention humaine dans le
                    processus décisionnel (human-in-the-loop), avant ou
                    après la décision. Quatrièmement, une obligation de
                    journalisation, d'enregistrement ou de traçabilité
                    technique du fonctionnement d'un système (boîte noire,
                    journal d'événements, registre d'audit), qu'elle soit
                    destinée à une autorité de contrôle, à un organe interne
                    ou au public. Cinquièmement, un droit de recours,
                    d'opposition ou de contestation contre une décision
                    prise par un système d'IA, ou une procédure de
                    réclamation propre à un tel système. Sixièmement, une
                    obligation de publier ou de communiquer des
                    caractéristiques techniques d'un système — sa
                    documentation technique, les données utilisées pour
                    l'entraîner, ses capacités, ses limites connues, son
                    architecture, ses paramètres, sa méthode d'évaluation ou
                    de certification — à un public, des utilisateurs, une
                    autorité de surveillance ou un organe de certification,
                    y compris une procédure de certification ou d'audit de
                    conformité qui impose une telle communication ou
                    vérification. Dans tous les cas, l'objet de la norme
                    doit être de rendre le fonctionnement ou la décision du
                    système SCRUTABLE — compréhensible, vérifiable,
                    contrôlable ou contestable — par une personne, une
                    autorité ou le public, quel que soit le secteur dans
                    lequel le système est utilisé.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible : une mesure qui interdit,
                    restreint, corrige ou sanctionne directement un résultat
                    produit par un système d'IA — une décision
                    discriminatoire, un contenu dangereux ou trompeur, une
                    action autonome dommageable — sans se limiter à en
                    assurer la traçabilité, l'explication ou la
                    contestabilité : une telle mesure agit sur le résultat
                    produit lui-même, elle ne se contente pas de le rendre
                    scrutable ; une exigence dont l'objet spécifique est de
                    protéger l'intégrité technique d'un système contre une
                    attaque, une intrusion ou une défaillance (sécurité,
                    robustesse, résilience), sans lien avec la possibilité
                    de comprendre, de tracer ou de contester son
                    fonctionnement ; une règle qui régule le traitement de
                    données personnelles en général (base légale, finalité,
                    durée de conservation) sans imposer d'obligation
                    d'information, d'explication, de traçabilité ou de
                    contestation spécifiquement liée à un système d'IA ; et
                    une obligation de transparence administrative générale
                    (accès aux documents, principe de publicité) sans lien
                    avec un système d'intelligence artificielle. Le simple
                    emploi des mots « transparence », « responsabilité », «
                    explicabilité » ou « traçabilité » ne suffit jamais : il
                    faut que le texte impose concrètement l'un des six
                    mécanismes décrits ci-dessus. En cas de doute,
                    demande-toi : cet article rend-il le fonctionnement,
                    l'usage ou la décision d'un système d'IA transparent,
                    traçable, explicable, contrôlable, contestable, ou
                    attribuable à un acteur responsable (OUI), ou agit-il
                    directement sur le résultat produit par le système,
                    protège-t-il son intégrité technique, ou régule-t-il les
                    données en général sans mécanisme de scrutabilité
                    spécifique à l'IA (NON) ? Ta justification doit citer
                    l'élément précis du texte qui organise cette
                    scrutabilité — jamais reformuler la définition de la
                    cible elle-même. Si tu ne peux pas pointer cet élément
                    précis, la réponse est NON.
                """),
                "examples": [
                    "Une obligation d'informer une personne qu'une décision administrative la concernant a été prise, en tout ou en partie, par un système d'intelligence artificielle.",
                    "Une obligation d'expliquer les motifs d'une décision de crédit rendue par un système de décision automatisée.",
                    "Une obligation de supervision humaine avant qu'une décision prise par un système d'intelligence artificielle ne devienne définitive.",
                    "Une obligation de journalisation des décisions prises par un système d'intelligence artificielle utilisé par une administration, destinée à un contrôle a posteriori.",
                    "Un droit de recours contre une décision individuelle rendue par un système de décision automatisée.",
                    "Une obligation faite aux fournisseurs de systèmes d'intelligence artificielle de publier une documentation technique décrivant le fonctionnement, les données d'entraînement et les limites connues de leur système.",
                    "Une procédure de certification ou d'audit de conformité vérifiant le respect d'exigences de transparence par un système d'intelligence artificielle avant sa mise sur le marché.",
                ],
            },
        ),
        (
            "OUTPUT_HARMS",
            {
                "name": "Préjudices liés aux résultats de l'IA",
                "prompt_label": "Préjudices causés par les décisions, contenus ou actions produits par un système d'intelligence artificielle",
                "quadrant": "Safeguarding x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme PRÉVIENT,
                    RESTREINT, CORRIGE ou offre un REMÈDE contre un résultat
                    concret produit DIRECTEMENT par un système
                    d'intelligence artificielle, lorsque ce résultat est
                    dommageable, illicite, dangereux, discriminatoire,
                    trompeur ou autrement problématique. Le résultat visé
                    peut prendre la forme d'un contenu généré (texte, image,
                    son, vidéo), d'une décision individuelle (refus, octroi,
                    évaluation, classification, sanction), d'une prédiction,
                    d'une recommandation, d'une commande ou d'une action
                    physique exécutée par un système autonome. Quatre
                    familles de mesures satisfont typiquement cette cible.
                    Premièrement, une interdiction ou une restriction visant
                    une décision automatisée discriminatoire ou une décision
                    individuelle dangereuse dans un contexte à hauts enjeux
                    pour les personnes (emploi, crédit, santé, éducation,
                    justice, migration, prestations sociales).
                    Deuxièmement, une exigence de sécurité visant à prévenir
                    un accident ou un dommage physique causé par l'action
                    autonome d'un robot, d'un véhicule ou d'une machine
                    pilotée par un système d'intelligence artificielle.
                    Troisièmement, une règle attribuant la titularité, la
                    protection ou la responsabilité pour un CONTENU produit
                    en sortie par un système d'IA — droit d'auteur sur une
                    création générée, contrefaçon d'une œuvre préexistante
                    par un contenu généré, usurpation d'identité, contenu
                    synthétique représentant une personne sans son
                    consentement (deepfake). Quatrièmement, une obligation
                    d'étiqueter, de signaler ou de corriger un contenu ou
                    une recommandation trompeurs, dangereux ou illicites
                    produits par un système d'IA. Dans tous les cas, l'objet
                    de la norme doit être le résultat lui-même — ce que le
                    système produit, décide ou fait —, et non son
                    fonctionnement interne ni la possibilité de le
                    comprendre ou de le contester.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible : une obligation
                    d'informer une personne qu'une décision la concernant a
                    été prise par un système d'IA, une obligation
                    d'expliquer les motifs d'une décision, une obligation de
                    supervision humaine, de traçabilité ou de publication de
                    caractéristiques techniques d'un système — ces
                    obligations rendent le fonctionnement du système
                    scrutable (compréhensible, vérifiable, contestable),
                    mais n'agissent pas elles-mêmes sur le résultat produit ;
                    une exigence dont l'objet est de protéger l'intégrité
                    technique d'un système contre une attaque, une intrusion
                    ou une défaillance, sans viser un résultat concret
                    dommageable déjà produit ou à produire ; une règle qui
                    régule des droits ou des conditions attachés à des
                    données ou contenus utilisés EN AMONT pour entraîner un
                    système — licéité, consentement, licence, exception au
                    droit d'auteur sur des données ou œuvres d'entraînement
                    — plutôt qu'à un contenu ou une décision produits en
                    sortie ; une règle de sécurité physique visant des
                    machines ou des robots dépourvus de toute capacité de
                    décision ou d'apprentissage autonome (mécanique ou
                    électronique classique, sans IA) ; et une régulation
                    générale des médias, de la publicité ou de la
                    désinformation sans lien structurel avec l'intelligence
                    artificielle ou l'automatisation. Le simple fait qu'un
                    contenu ou une décision puisse théoriquement provenir
                    d'un système d'IA ne suffit pas : il faut que le texte
                    vise spécifiquement un résultat produit par un système
                    d'IA ou automatisé. En cas de doute, demande-toi : cet
                    article prévient-il, restreint-il, corrige-t-il ou
                    offre-t-il un remède contre un résultat concret —
                    contenu, décision, prédiction, recommandation, action —
                    directement produit par un système d'IA (OUI), ou
                    s'agit-il de rendre le système scrutable, de protéger
                    son intégrité technique, de réguler ses données
                    d'entraînement, ou d'une régulation générale sans lien
                    spécifique avec l'IA (NON) ? Ta justification doit citer
                    l'élément précis du texte qui vise ce résultat produit —
                    jamais reformuler la définition de la cible elle-même.
                    Si tu ne peux pas pointer cet élément précis, la réponse
                    est NON.
                """),
                "examples": [
                    "Une interdiction d'utiliser un système d'intelligence artificielle pour prendre seul une décision de refus de crédit.",
                    "Un encadrement spécifique des décisions automatisées évaluant l'éligibilité à des prestations sociales.",
                    "Une obligation de sécurité applicable à des robots ou véhicules pilotés par un système d'intelligence artificielle, destinée à prévenir le risque d'accident pour les personnes à proximité.",
                    "Une règle attribuant la responsabilité en cas de contrefaçon lorsqu'un contenu produit par un système d'intelligence artificielle reproduit une œuvre protégée préexistante.",
                    "Une obligation d'étiqueter un contenu généré ou modifié par intelligence artificielle (deepfake) représentant une personne réelle sans son consentement.",
                    "Une règle excluant de la protection par le droit d'auteur une œuvre entièrement générée par une intelligence artificielle, faute d'auteur humain identifiable.",
                ],
            },
        ),
        (
            "SOCIETAL_HARMS",
            {
                "name": "Préjudices sociétaux",
                "prompt_label": "Préjudices sociétaux et systémiques liés à la diffusion de l'intelligence artificielle",
                "quadrant": "Safeguarding x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme s'attaque à une
                    conséquence COLLECTIVE, SYSTÉMIQUE ou SOCIÉTALE de
                    l'utilisation généralisée de l'intelligence artificielle
                    — un effet qui touche la société, les institutions, les
                    marchés, les processus démocratiques, l'environnement
                    informationnel ou les rapports sociaux dans leur
                    ensemble, plutôt qu'une décision ou un résultat
                    individuel isolé. Cela recouvre notamment : la
                    désinformation ou la manipulation de l'opinion publique
                    à grande échelle rendue possible par la génération, la
                    recommandation ou la diffusion automatisée massive de
                    contenu ; l'atteinte aux processus électoraux ou
                    démocratiques par un usage coordonné ou massif de
                    systèmes d'intelligence artificielle ; une
                    discrimination structurelle ou systémique résultant de
                    l'usage généralisé de systèmes d'IA dans un secteur ou
                    dans la société, distincte d'une décision discriminatoire
                    individuelle isolée ; la dégradation de l'environnement
                    informationnel (perte de confiance dans l'information,
                    saturation par du contenu synthétique) ; ou des risques
                    institutionnels ou de marché plus larges liés à une
                    adoption massive de l'IA (concentration excessive du
                    marché, dépendance systémique, instabilité). L'objet de
                    la norme doit être cet effet AGRÉGÉ ou COLLECTIF, et non
                    un dommage individuel isolé causé à une personne
                    déterminée par une décision ou un contenu particulier.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible : une interdiction ou une
                    restriction visant une décision, un contenu ou une
                    action individuelle produits par un système d'IA — une
                    décision discriminatoire prise à l'égard d'une personne
                    déterminée, un contenu illicite isolé, un accident
                    causé par un système autonome — même si ce type de
                    résultat pourrait, en théorie, se reproduire à grande
                    échelle : tant que l'objet de la norme reste la
                    protection d'une personne déterminée contre un résultat
                    individuel, la mesure ne satisfait pas cette cible ; une
                    obligation d'informer une personne, d'expliquer une
                    décision, ou de publier des caractéristiques techniques
                    d'un système, qui rend le système scrutable sans
                    s'attaquer à un effet collectif ou systémique ; une
                    exigence de protection technique d'un système contre une
                    attaque ou une défaillance ; une règle régulant des
                    droits ou des conditions attachés à des données
                    d'entraînement ; et une régulation générale des médias,
                    de la publicité, de la désinformation ou de la
                    concurrence économique sans lien structurel avec
                    l'intelligence artificielle ou l'automatisation — le
                    simple fait qu'un média ou un marché puisse en théorie
                    être affecté par l'IA ne suffit pas, il faut que le
                    texte vise spécifiquement les effets collectifs de
                    systèmes d'IA ou de traitements automatisés. En cas de
                    doute, demande-toi : cet article s'attaque-t-il à un
                    effet AGRÉGÉ, COLLECTIF ou SYSTÉMIQUE de l'usage
                    généralisé de l'IA sur la société, les institutions, les
                    marchés ou les processus démocratiques (OUI), ou
                    s'agit-il de protéger une personne déterminée contre un
                    résultat individuel, de rendre un système scrutable, de
                    le protéger techniquement, ou d'une régulation générale
                    sans lien spécifique avec l'IA (NON) ? Ta justification
                    doit citer l'élément précis du texte qui vise cet effet
                    collectif — jamais reformuler la définition de la cible
                    elle-même. Si tu ne peux pas pointer cet élément précis,
                    la réponse est NON.
                """),
                "examples": [
                    "Une règle limitant l'usage de systèmes de recommandation automatisés qui amplifient la désinformation à grande échelle.",
                    "Une interdiction de l'usage de systèmes d'intelligence artificielle pour manipuler le comportement électoral de façon coordonnée et massive.",
                    "Une obligation de surveiller ou d'atténuer les risques de discrimination systémique liés à l'usage généralisé de systèmes d'intelligence artificielle dans un secteur.",
                    "Une mesure visant à préserver la diversité et la fiabilité de l'environnement informationnel face à la diffusion massive de contenu généré par intelligence artificielle.",
                    "Une règle visant à prévenir une concentration excessive de marché ou une dépendance systémique résultant de l'adoption massive de systèmes d'intelligence artificielle.",
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
    label = d["prompt_label"]
    examples = "\n".join(f"- {ex}" for ex in d["examples"])

    blocks = [
        "Tu es un expert en analyse des politiques publiques et du droit suisse.",
        f"## Contexte : l'intelligence artificielle et sa régulation\n\n{_AI_CONTEXT}",
        f"## Cible à évaluer : {label}\n\n{d['definition']}\n\n"
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
