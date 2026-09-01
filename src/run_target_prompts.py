# src/run_target_prompts.py
# Un prompt système dédié par cible (target), sur le même modèle que
# run_inst_prompts.py : une question binaire fermée par cible, au lieu d'une
# seule question générique couvrant les 10 cibles à la fois.
#
# Taxonomie des 10 cibles opérationnelles (fusion des deux cibles Safeguarding
# x Downstream "Usages à hauts enjeux & Droits fondamentaux" et "Information &
# Préjudices sociétaux" en une seule cible "Risques à hauts enjeux") : PoC
# PDF, section 2.3 (nouvelle grille à 4 quadrants Enabling/Safeguarding x
# Upstream/Downstream). Cette
# taxonomie remplace celle de run5_prompts.TARGET_CODES (10 cibles, ancienne
# version) — ne pas réutiliser run5_prompts pour ce pipeline.
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
    la vie privée, atteintes aux droits de propriété intellectuelle,
    défaillances de sécurité, décisions inexplicables ou incontestables,
    conséquences lourdes pour les droits fondamentaux, ou préjudices
    sociétaux liés à la diffusion automatisée d'information.
""")

# ---------------------------------------------------------------------------
# Les 10 cibles opérationnelles, dans l'ordre du tableau 2.3 du PDF.
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
                "quadrant": "Enabling x Upstream",
                "definition": _p("""
                    Cette cible répond à une question simple : cette norme
                    permet-elle à des développeurs de systèmes
                    d'intelligence artificielle — les acteurs, publics ou
                    privés, qui construisent ou entraînent des modèles —
                    d'utiliser des données (collecte, moissonnage/scraping,
                    ou réutilisation de contenus) pour entraîner leurs
                    modèles ? Deux cas satisfont cette cible. Cas explicite
                    : la norme porte spécifiquement sur les données
                    d'entraînement de systèmes d'IA — par exemple des jeux
                    de données mis à disposition pour entraîner des
                    modèles, ou un cadre facilitant le partage de données à
                    cette fin. Cas implicite : la norme retire une
                    protection — typiquement le droit d'auteur — qui aurait
                    autrement empêché la réutilisation d'une donnée ou d'une
                    œuvre, ce qui la rend de fait librement utilisable par
                    quiconque veut entraîner un modèle d'IA, sans que le
                    texte ait besoin de mentionner l'IA. Une règle qui fixe
                    de façon générale le champ d'application d'une loi sur
                    la protection des données — déterminant quels
                    traitements, y compris le scraping par des tiers, y
                    sont soumis — relève aussi de ce second cas.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible les normes qui donnent à un
                    organe de l'État — autorité de surveillance, police,
                    service de renseignement, assureur public, régulateur
                    sectoriel — un accès ou un pouvoir de traitement sur des
                    données pour l'accomplissement de SES propres tâches.
                    Dans ce contexte, l'État n'est pas un développeur de
                    systèmes d'IA : une telle habilitation ne rend aucune
                    donnée plus accessible aux développeurs de modèles,
                    même si le texte emploie des mots comme « traiter des
                    données », « données sensibles » ou « accès aux
                    données ». Si le texte ne relève d'aucun des deux cas
                    positifs décrits plus haut, la réponse est NON.
                """),
                "examples": [
                    "Une obligation faite aux administrations de mettre à disposition des jeux de données ouvertes utilisables pour entraîner des systèmes d'intelligence artificielle.",
                    "Un cadre légal facilitant le partage de données de santé anonymisées à des fins de recherche en intelligence artificielle.",
                    "Une exception au droit d'auteur qui exclut les lois, ordonnances et actes officiels de la protection du droit d'auteur, rendant ces textes librement réutilisables par quiconque.",
                    "Une exception au droit d'auteur autorisant la reproduction d'œuvres pour l'usage privé ou la documentation interne, y compris une exception pour la fouille de textes et de données sur des œuvres déjà licitement accessibles.",
                    "Une exception au droit d'auteur autorisant la reproduction de courts extraits d'articles de presse à des fins d'information sur l'actualité.",
                    "L'article qui fixe le champ d'application général d'une loi sur la protection des données et détermine quels traitements de données, y compris automatisés, sont soumis à la surveillance de l'autorité compétente.",
                ],
            },
        ),
        (
            "COMPUTE_INFRASTRUCTURE",
            {
                "name": "Calcul & Infrastructure",
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
                    eau, transport, télécoms) dont l'objet est la protection
                    de ces infrastructures elles-mêmes, pas le calcul pour
                    l'IA — une telle protection relève, le cas échéant, de
                    la cible Sécurité & Robustesse, pas de celle-ci ; ou une
                    règle générale de gouvernance ou de sécurité des
                    infrastructures numériques de l'État (systèmes
                    d'information, cybersécurité administrative) sans
                    rapport avec la fourniture de capacité de calcul pour
                    l'IA. Le simple emploi des mots « infrastructure »,
                    « numérique », « centre de données », « énergie » ou
                    « calcul » ne suffit jamais : il faut que le texte relie
                    explicitement cette ressource au développement ou au
                    fonctionnement de systèmes d'intelligence artificielle.
                    En cas de doute, demande-toi : cet article finance-t-il,
                    construit-il ou facilite-t-il concrètement l'accès à une
                    capacité de calcul (puces, cloud, supercalcul, centre de
                    données, énergie dédiée) destinée à l'entraînement ou au
                    fonctionnement de systèmes d'IA (OUI), ou s'agit-il
                    d'infrastructure, d'énergie, de numérique ou de
                    télécommunications en général, sans lien spécifique avec
                    le calcul pour l'IA (NON) ? Un rapprochement lexical
                    avec le mot « infrastructure » ne suffit jamais : ta
                    justification doit citer l'élément précis du texte qui
                    finance ou facilite l'accès à une capacité de calcul
                    pour l'IA — jamais reformuler la définition de la cible
                    elle-même. Si tu ne peux pas pointer cet élément précis,
                    la réponse est NON.
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
            "ADOPTION_DIFFUSION",
            {
                "name": "Adoption & Diffusion",
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
                    dérogatoire — qui PERMET à une entreprise, une
                    administration ou une autre organisation d'utiliser ou
                    de tester un système d'intelligence artificielle en
                    conditions réelles, même de façon limitée dans le temps,
                    l'espace ou le champ d'application, avant sa
                    généralisation : en autorisant cet usage réel, l'État
                    encourage directement l'adoption et la diffusion du
                    système, tout comme dans les trois cas précédents. Dans
                    les quatre cas, c'est l'UTILISATION du système d'IA par
                    son destinataire final — l'entreprise, l'administration
                    ou l'organisation qui s'en sert, même à titre pilote ou
                    expérimental — qui doit être directement encouragée,
                    financée, autorisée ou facilitée par la norme ; peu
                    importe le secteur (santé, justice, transport, etc.) :
                    ce qui compte est que la mesure vise l'usage réel de
                    l'IA par son destinataire final, pas un autre maillon
                    de la chaîne.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible les mesures qui, bien que
                    liées à l'intelligence artificielle, portent en
                    réalité sur un autre maillon de la chaîne de valeur ou
                    relèvent d'une autre logique de régulation — même si,
                    par ricochet, elles pourraient elles aussi favoriser
                    l'IA en général. Ne satisfont donc pas cette cible : un
                    financement de la recherche ou de l'innovation en
                    intelligence artificielle qui ne finance et n'impose
                    aucun usage concret par une entreprise ou une
                    administration (cible Recherche & Innovation) ; un
                    dispositif de formation, d'enseignement ou de
                    développement de compétences en IA, qui outille des
                    personnes sans encourager ni faciliter l'utilisation
                    d'un système par une organisation (cible Compétences &
                    Capital humain) ; une mesure facilitant l'accès aux
                    données d'entraînement ou à la puissance de calcul, qui
                    profite aux développeurs de systèmes d'IA et non à
                    leurs utilisateurs finaux (cibles Accès aux données &
                    Ressources et Calcul & Infrastructure) ; ainsi que toute
                    exigence de sécurité, de robustesse, de transparence, de
                    traçabilité, de supervision humaine, de protection des
                    données, de protection de droits fondamentaux ou de
                    protection contre des préjudices sociétaux, même
                    lorsque cette exigence porte sur un système d'IA déjà
                    utilisé ou déployé, y compris dans le cadre d'un bac à
                    sable ou d'un essai pilote — une telle exigence encadre
                    ou restreint l'usage, elle ne l'encourage ni ne le
                    facilite (cibles relevant de la logique Safeguarding).
                    Une simple mention de la « transformation numérique »,
                    de la « modernisation de l'administration » ou de la «
                    digitalisation », sans référence spécifique à
                    l'intelligence artificielle, ne satisfait pas non plus
                    cette cible. En cas de doute, demande-toi : cet article
                    encourage-t-il, finance-t-il, autorise-t-il ou
                    facilite-t-il concrètement l'UTILISATION ou le
                    DÉPLOIEMENT — y compris à titre pilote ou expérimental —
                    d'un système d'intelligence artificielle par une
                    entreprise, une administration ou une organisation qui
                    s'en sert dans son activité (OUI), ou s'agit-il d'autre
                    chose — recherche, formation, données, calcul,
                    sécurité, gouvernance, protection des droits (NON) ? Un
                    rapprochement lexical avec les mots « adoption », «
                    diffusion », « déploiement », « pilote », « bac à
                    sable » ou « utilisation » ne suffit jamais : ta
                    justification doit citer l'élément précis du texte qui
                    encourage, finance, autorise ou facilite concrètement
                    l'usage d'un système d'IA par son destinataire final —
                    jamais reformuler la définition de la cible elle-même.
                    Si tu ne peux pas pointer cet élément précis, la réponse
                    est NON.
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
            "DATA_PRIVACY",
            {
                "name": "Données & Vie privée",
                "quadrant": "Safeguarding x Upstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme protège des
                    droits ou intérêts liés au traitement de données
                    personnelles ou de données privées (collecte,
                    conservation, réutilisation, communication à des tiers,
                    croisement, profilage, ré-identification, inférence),
                    selon deux cas. Premier cas — régime général : la norme
                    régule de façon GÉNÉRALE le traitement de données
                    personnelles, quel que soit le moyen utilisé pour le
                    traiter (base légale requise, finalité, principe de
                    proportionnalité, durée de conservation, droit d'accès
                    ou de rectification, obligation de sécurité des
                    données) ; un tel régime général s'applique par
                    construction aussi bien au traitement par un système
                    d'intelligence artificielle ou par un système automatisé
                    qu'à tout autre traitement, même si le texte ne
                    mentionne jamais l'IA. Second cas — régime spécifique :
                    la norme régule SPÉCIFIQUEMENT le traitement de données
                    personnelles PAR un système d'intelligence artificielle
                    ou par un système automatisé de traitement de données —
                    par exemple une exigence propre à l'usage de l'IA pour
                    traiter, croiser, profiler ou ré-identifier des
                    personnes à partir de données. Dans les deux cas, la
                    mesure porte sur l'UTILISATION ou le TRAITEMENT de
                    données par un système d'intelligence artificielle déjà
                    développé ou par un système automatisé déjà en fonction
                    — pas sur les données utilisées pour ENTRAÎNER un tel
                    système ; une mesure qui porte spécifiquement sur les
                    données d'entraînement d'un système d'IA relève d'une
                    autre cible, pas de celle-ci.
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
                    traitées ; et une mesure portant spécifiquement sur les
                    données d'entraînement d'un système d'intelligence
                    artificielle (anonymisation avant entraînement, droit de
                    savoir quelles données ont servi à l'entraînement,
                    licéité de la réutilisation d'un jeu de données pour
                    entraîner un modèle), qui relève d'une autre cible que
                    celle-ci. Le simple emploi des mots « données », « vie
                    privée », « protection des données » ou « traitement »
                    ne suffit jamais : il faut que le texte régule un
                    traitement de données au sens de la définition ci-dessus
                    — de façon générale, ou spécifiquement par l'IA — et non
                    un simple pouvoir d'accès, une génération de données non
                    automatisée, ou un régime propre aux données
                    d'entraînement. En cas de doute, demande-toi : cet
                    article régule-t-il, de façon générale ou spécifiquement
                    pour l'IA, le TRAITEMENT (collecte, conservation,
                    réutilisation, croisement, ré-identification) de données
                    personnelles déjà collectées, dans le cadre de
                    l'UTILISATION d'un système d'IA ou d'un système
                    automatisé (OUI), ou s'agit-il d'un pouvoir d'accès
                    accordé à une administration, d'une génération de
                    données par un processus non automatisé, ou d'un régime
                    propre aux données d'entraînement (NON) ? Ta
                    justification doit citer l'élément précis du texte qui
                    régule ce traitement — jamais reformuler la définition
                    de la cible elle-même. Si tu ne peux pas pointer cet
                    élément précis, la réponse est NON.
                """),
                "examples": [
                    "Une règle générale fixant les conditions de licéité (base légale, finalité, proportionnalité) de tout traitement de données personnelles, applicable aussi bien à un traitement manuel qu'à un traitement par un système automatisé ou par une intelligence artificielle.",
                    "Une obligation de pseudonymiser ou d'anonymiser des données personnelles avant leur traitement par un système d'intelligence artificielle utilisé à des fins de profilage ou de classification de personnes.",
                    "Une interdiction de ré-identifier une personne à partir de données que l'intelligence artificielle a rendues anonymes.",
                    "Un droit de s'opposer au traitement de ses données personnelles par un système de décision automatisée ou d'intelligence artificielle.",
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
                    Cette cible est satisfaite si la norme régule des droits
                    de propriété intellectuelle ou des droits d'auteur
                    portant spécifiquement sur le CONTENU produit EN SORTIE
                    (output) par un système d'intelligence artificielle ou
                    par un traitement automatisé — texte, image, son, vidéo,
                    code, ou toute autre création générée de façon autonome
                    ou semi-autonome par un tel système. Cela recouvre
                    notamment : la question de savoir si un contenu généré
                    par IA peut être protégé par le droit d'auteur, et à
                    quelles conditions (par exemple l'exigence d'un apport
                    créatif humain) ; l'attribution de la titularité des
                    droits sur un tel contenu — à son utilisateur, au
                    développeur ou à l'exploitant du système, ou à personne
                    ; l'exercice de droits moraux sur un contenu généré par
                    IA ; les conditions d'exploitation ou de commercialisation
                    d'un contenu généré par IA ; ou la responsabilité
                    encourue lorsqu'un contenu généré par un système
                    d'intelligence artificielle reproduit ou imite une œuvre
                    protégée préexistante (contrefaçon). Dans tous les cas,
                    c'est le contenu produit en sortie par le système qui
                    doit être l'objet de la règle.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible les normes qui portent sur
                    des données ou des œuvres utilisées EN AMONT pour
                    entraîner un système d'intelligence artificielle — jeux
                    de données d'entraînement, œuvres protégées moissonnées
                    ou réutilisées pour l'entraînement, exception au droit
                    d'auteur pour la fouille de textes et de données (text
                    and data mining), ou obligation de rémunérer des
                    titulaires de droits dont les œuvres ont servi à
                    l'entraînement d'un modèle. Ces mesures régulent
                    l'INPUT du système, pas son output : elles relèvent de
                    la cible Accès aux données & Ressources, pas de
                    celle-ci, même si elles emploient un vocabulaire de
                    propriété intellectuelle ou de droit d'auteur. Ne
                    satisfont pas non plus cette cible les règles générales
                    de propriété intellectuelle, de droit d'auteur ou de
                    propriété industrielle (brevets, marques, dessins et
                    modèles) qui ne visent pas spécifiquement un contenu
                    généré par un système automatisé ou d'intelligence
                    artificielle — par exemple la durée de protection d'une
                    œuvre, une exception pour l'usage privé ou
                    l'accessibilité aux personnes handicapées, ou
                    l'organisation d'une autorité chargée de la propriété
                    intellectuelle en général. Le simple emploi des mots «
                    propriété intellectuelle », « droit d'auteur », «
                    création » ou « contenu » ne suffit jamais : il faut que
                    le texte régule spécifiquement le contenu produit en
                    sortie par un système d'IA. En cas de doute, demande-toi
                    : cet article régule-t-il un droit de propriété
                    intellectuelle ou d'auteur sur un contenu généré EN
                    SORTIE par un système d'intelligence artificielle (OUI),
                    ou porte-t-il sur les données utilisées pour ENTRAÎNER
                    un tel système, ou sur la propriété intellectuelle en
                    général sans lien avec un contenu généré par IA (NON) ?
                    Un rapprochement lexical avec le vocabulaire de la
                    propriété intellectuelle ne suffit jamais : ta
                    justification doit citer l'élément précis du texte qui
                    régule ce contenu généré en sortie — jamais reformuler
                    la définition de la cible elle-même. Si tu ne peux pas
                    pointer cet élément précis, la réponse est NON.
                """),
                "examples": [
                    "Une règle attribuant les droits d'auteur sur un contenu généré par un système d'intelligence artificielle.",
                    "Une disposition excluant de la protection par le droit d'auteur une œuvre entièrement générée par une intelligence artificielle, faute d'auteur humain identifiable.",
                    "Une règle attribuant la responsabilité en cas de contrefaçon lorsqu'un contenu produit par un système d'intelligence artificielle reproduit une œuvre protégée préexistante.",
                    "Une obligation d'obtenir l'autorisation d'un ayant droit avant d'exploiter commercialement un contenu généré par intelligence artificielle imitant le style ou la voix d'un artiste.",
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
                    exigence dont l'objet spécifique est de protéger un
                    système, un modèle, des données d'entraînement ou de
                    fonctionnement, ou une infrastructure numérique
                    d'intelligence artificielle — ou un système automatisé
                    de traitement de données — contre une CYBERATTAQUE :
                    une intrusion, un accès non autorisé, un piratage, une
                    manipulation malveillante (par exemple un empoisonnement
                    de données ou une attaque adverse visant à tromper un
                    modèle), un sabotage, ou toute autre atteinte à la
                    confidentialité, à l'intégrité ou à la disponibilité
                    d'un tel système provoquée par un acteur malveillant.
                    Peu importe que le système ou l'infrastructure visé
                    appartienne à l'État (administration, serveur public)
                    ou à un acteur privé (entreprise développant ou
                    exploitant un système d'IA, opérateur d'une
                    infrastructure critique) : ce qui compte est que la
                    mesure protège spécifiquement un système d'IA ou un
                    système automatisé contre un risque cyber.
                """) + " " + _p("""
                    Ne satisfont PAS cette cible : une mesure qui protège la
                    confidentialité de données personnelles ou régule
                    généralement leur traitement (base légale, finalité,
                    durée de conservation, droit d'accès ou de
                    rectification) — même lorsqu'elle emploie le mot «
                    sécurité » (par exemple une « obligation de sécurité des
                    données ») ; une telle obligation relève de la cible
                    Données & Vie privée, pas de celle-ci, sauf si elle vise
                    spécifiquement à empêcher une intrusion ou un accès
                    malveillant à un système d'IA ou automatisé. Ne
                    satisfont pas non plus cette cible une exigence dont
                    l'objet est la protection de droits fondamentaux, la
                    non-discrimination, l'explicabilité, la transparence, la
                    supervision humaine ou la possibilité de contester une
                    décision — ces exigences relèvent des cibles
                    Responsabilité & Transparence ou Usages à hauts enjeux &
                    Droits fondamentaux, pas de celle-ci. Une exigence de
                    robustesse, de fiabilité ou d'exactitude technique d'un
                    système d'IA qui n'est pas motivée par un risque
                    d'attaque ou d'intrusion malveillante — par exemple un
                    test de performance, une exigence de qualité des données
                    d'entraînement, une exigence de résistance à des données
                    aberrantes ou à des erreurs non malveillantes, ou une
                    obligation générale de fiabilité d'un service numérique
                    — ne satisfait pas non plus cette cible : sans lien avec
                    une menace malveillante (piratage, intrusion, sabotage),
                    il ne s'agit pas de sécurité au sens de cette cible. Ne
                    satisfont pas non plus cette cible une exigence de
                    sécurité physique (bâtiment, coffre-fort, protection
                    périmétrique), une loi ou une stratégie générale de
                    cybersécurité de l'État ou d'une entreprise sans lien
                    spécifique avec un système d'IA ou un système automatisé
                    de traitement de données, ou une définition ou un
                    inventaire des infrastructures critiques (énergie, eau,
                    transport, télécoms) dont l'objet est la protection de
                    ces infrastructures en général, sans lien spécifique
                    avec un système d'IA ou un système automatisé qui les
                    pilote ou les traite. Le simple emploi des mots «
                    sécurité », « robustesse », « résilience » ou «
                    cybersécurité » ne suffit jamais : il faut que le texte
                    protège spécifiquement, contre une menace malveillante,
                    un système d'IA ou un système automatisé de traitement
                    de données. En cas de doute, demande-toi : cet article
                    impose-t-il une exigence dont l'objet spécifique est de
                    protéger un système d'IA ou un système automatisé contre
                    une cyberattaque — intrusion, piratage, manipulation
                    malveillante, sabotage (OUI) — ou s'agit-il de
                    protection des données personnelles, de droits
                    fondamentaux, de robustesse technique générale sans lien
                    avec une menace malveillante, ou de cybersécurité
                    générale sans lien spécifique avec l'IA (NON) ? Un
                    rapprochement lexical avec le vocabulaire de la
                    cybersécurité ne suffit jamais : ta justification doit
                    citer l'élément précis du texte qui protège un système
                    d'IA ou automatisé contre une cyberattaque — jamais
                    reformuler la définition de la cible elle-même. Si tu ne
                    peux pas pointer cet élément précis, la réponse est NON.
                """),
                "examples": [
                    "Une obligation de certification de cybersécurité pour les systèmes d'intelligence artificielle utilisés dans une infrastructure critique, destinée à prévenir les intrusions et le piratage.",
                    "Une exigence de test de résistance aux attaques adverses (adversarial attacks) ou à l'empoisonnement de données avant la mise en service d'un système d'intelligence artificielle dans un contexte sensible.",
                    "Une obligation pour les fournisseurs de systèmes d'intelligence artificielle de mettre en place des mesures de protection contre l'accès non autorisé ou le piratage de leurs modèles et infrastructures.",
                    "Une obligation pour les administrations publiques de sécuriser leurs serveurs et systèmes automatisés de traitement de données contre les cyberattaques.",
                    "Une exigence de détection et de notification des incidents de cybersécurité affectant un système d'intelligence artificielle déployé par une entreprise ou une administration.",
                ],
            },
        ),
        (
            "ACCOUNTABILITY_TRANSPARENCY",
            {
                "name": "Responsabilité & Transparence",
                "quadrant": "Safeguarding x Downstream",
                "definition": _p("""
                    Cette cible est étroite et ne couvre que deux cas
                    précis. Premier cas : la norme reconnaît ou impose
                    explicitement, au bénéfice d'une personne concernée, un
                    droit ou une obligation d'INFORMER cette personne
                    qu'une décision la concernant directement (refus,
                    octroi, évaluation, classification, sanction) a été
                    prise, en tout ou en partie, par un système
                    d'intelligence artificielle ou par un traitement
                    automatisé de données. Second cas : la norme impose aux
                    développeurs, fournisseurs ou exploitants d'un système
                    d'intelligence artificielle une obligation de rendre
                    publics ou de communiquer des ASPECTS TECHNIQUES de leur
                    système — son fonctionnement, les données utilisées
                    pour l'entraîner, ses capacités, ses limites connues,
                    son architecture, ses paramètres, sa méthode
                    d'évaluation ou de test — que le destinataire de cette
                    communication soit le public, les utilisateurs, une
                    autorité de surveillance ou un organe de certification.
                    Dans les deux cas, l'objet précis de la norme doit être
                    de FAIRE SAVOIR quelque chose à quelqu'un — que l'IA est
                    intervenue dans une décision, ou comment un système
                    d'IA fonctionne techniquement — pas d'organiser un autre
                    mécanisme de responsabilisation.
                """) + " " + _p("""
                    Ne satisfont donc PAS cette cible, même quand elles
                    relèvent d'une logique de responsabilisation de l'IA :
                    un droit de recours, d'opposition, de contestation ou
                    d'appel contre une décision, qui organise une voie de
                    contestation et non une obligation d'informer — sauf si
                    le même article impose aussi, distinctement,
                    d'informer la personne de l'intervention de l'IA, auquel
                    cas seule cette partie-là satisfait la cible ; une
                    obligation de supervision humaine ou de validation
                    humaine dans la boucle décisionnelle (human-in-the-loop,
                    intervention humaine avant ou après la décision), qui
                    organise un contrôle et non une communication
                    d'information ; une obligation de traçabilité, de
                    journalisation ou d'enregistrement interne (boîte noire,
                    journal d'événements, registre d'audit) destinée à une
                    autorité de contrôle a posteriori, tant qu'elle
                    n'impose aucune communication de documentation
                    technique ni aucune information de la personne
                    concernée ; une obligation d'explication détaillée des
                    motifs ou du raisonnement d'une décision individuelle
                    (explicabilité au sens strict), lorsque la norme ne se
                    limite pas à faire savoir qu'un système d'IA est
                    intervenu dans cette décision ; une procédure de
                    certification, d'audit de conformité ou d'évaluation
                    des risques d'un système d'IA qui n'impose elle-même
                    aucune obligation de publier ou de communiquer les
                    caractéristiques techniques du système ; une obligation
                    de transparence administrative générale (accès aux
                    documents, principe de publicité) sans lien avec un
                    système d'intelligence artificielle ; et tout
                    encadrement sectoriel spécifique de l'usage de l'IA
                    dans un contexte à hauts enjeux (santé, emploi, crédit,
                    justice, migration) ou toute exigence de sécurité, de
                    robustesse ou de non-discrimination, qui relèvent
                    d'autres cibles, pas de celle-ci, sauf si l'article
                    impose aussi, spécifiquement, l'un des deux cas décrits
                    ci-dessus. Le simple emploi des mots « transparence », «
                    responsabilité », « explicabilité » ou « traçabilité »
                    ne suffit jamais : il faut que le texte impose
                    concrètement soit une information de la personne
                    concernée sur l'intervention de l'IA dans une décision,
                    soit une communication des caractéristiques techniques
                    d'un système d'IA. En cas de doute, demande-toi : cet
                    article reconnaît-il ou impose-t-il explicitement (a)
                    un droit ou une obligation d'informer une personne
                    qu'une décision la concernant a été prise par un
                    système d'IA, ou (b) une obligation pour les
                    développeurs ou exploitants de communiquer des
                    informations techniques sur leur système d'IA (OUI), ou
                    s'agit-il d'autre chose — contestation, supervision
                    humaine, traçabilité interne, explication des motifs,
                    certification, transparence administrative générale,
                    encadrement sectoriel, sécurité, non-discrimination
                    (NON) ? Un rapprochement lexical avec le vocabulaire de
                    la transparence ou de la responsabilité ne suffit
                    jamais : ta justification doit citer l'élément précis
                    du texte qui informe une personne ou communique des
                    caractéristiques techniques — jamais reformuler la
                    définition de la cible elle-même. Si tu ne peux pas
                    pointer cet élément précis, la réponse est NON.
                """),
                "examples": [
                    "Une obligation d'informer une personne qu'une décision administrative la concernant a été prise, en tout ou en partie, par un système d'intelligence artificielle.",
                    "Un droit, pour une personne faisant l'objet d'une décision de crédit automatisée, d'être informée que cette décision a été rendue par un système d'intelligence artificielle.",
                    "Une obligation faite aux fournisseurs de systèmes d'intelligence artificielle de publier une documentation technique décrivant le fonctionnement, les données d'entraînement et les limites connues de leur système.",
                    "Une obligation de communiquer à une autorité de surveillance les caractéristiques techniques (architecture, paramètres, méthode d'évaluation) d'un système d'intelligence artificielle avant sa mise sur le marché.",
                ],
            },
        ),
        (
            "HIGH_STAKES_RISKS",
            {
                "name": "Risques à hauts enjeux",
                "quadrant": "Safeguarding x Downstream",
                "definition": _p("""
                    Cette cible est satisfaite si la norme régule
                    DIRECTEMENT l'utilisation d'un système d'intelligence
                    artificielle dans un contexte concret, dans le but d'en
                    réduire un risque pour les personnes ou pour la
                    société. Cela recouvre deux grandes familles de
                    risques. D'une part, les risques pour les droits et la
                    sécurité des individus lorsqu'un système d'IA est
                    utilisé dans un contexte à hauts enjeux : mobilité,
                    emploi, crédit, santé, éducation, police, justice,
                    prestations sociales, migration, discrimination, ou
                    encore le pilotage de machines ou de robots autonomes
                    dont un dysfonctionnement pourrait causer un accident
                    ou un dommage physique. D'autre part, les risques
                    sociétaux liés à l'utilisation de l'IA comme vecteur
                    d'information — génération, recommandation, ciblage ou
                    diffusion automatisée de contenu — susceptibles
                    d'amplifier la désinformation, de manipuler l'opinion
                    publique ou de fragiliser les processus démocratiques.
                    Dans les deux cas, l'objet de la norme doit être
                    l'utilisation elle-même du système d'IA dans ce
                    contexte à risque, et la mesure doit chercher à
                    réduire, prévenir ou encadrer ce risque (interdiction,
                    restriction, condition, garantie procédurale propre à
                    ce contexte).
                """) + " " + _p("""
                    Ne satisfont PAS cette cible : une obligation
                    d'informer une personne qu'une décision la concernant a
                    été prise par un système d'IA, ou une obligation de
                    publier ou communiquer des caractéristiques techniques
                    d'un système (documentation, architecture, données
                    d'entraînement, méthode d'évaluation) — ces
                    obligations, dont l'objet est de FAIRE SAVOIR quelque
                    chose plutôt que de réguler directement l'usage à
                    risque lui-même, relèvent de la cible Responsabilité &
                    Transparence, pas de celle-ci, même lorsqu'elles
                    s'appliquent dans un contexte à hauts enjeux ou
                    informationnel. Ne satisfont pas non plus cette cible
                    une exigence dont l'objet est de protéger un système
                    d'IA contre une cyberattaque (intrusion, piratage,
                    sabotage), qui relève de la cible Sécurité &
                    Robustesse, ni une règle de protection des données
                    personnelles ou de la vie privée, qui relève de la
                    cible Données & Vie privée — même si le traitement de
                    données a lieu dans un contexte à hauts enjeux ou
                    informationnel. Une règle de sécurité physique visant
                    des machines ou des robots dépourvus de toute capacité
                    de décision ou d'apprentissage autonome (mécanique ou
                    électronique classique, sans IA) ne satisfait pas non
                    plus cette cible : il faut que le risque visé provienne
                    spécifiquement d'un système d'IA. De même, une
                    régulation générale des médias, de la publicité, de la
                    désinformation, ou de la sécurité des machines et des
                    véhicules, sans lien structurel avec l'intelligence
                    artificielle ou l'automatisation, ne satisfait pas
                    cette cible : le simple fait qu'un contenu ou une
                    machine puisse théoriquement impliquer de l'IA ne
                    suffit pas, il faut que le texte vise spécifiquement
                    les systèmes d'IA ou les systèmes automatisés. En cas
                    de doute, demande-toi : cet article régule-t-il
                    directement l'utilisation d'un système d'IA dans un
                    contexte à hauts enjeux (droits fondamentaux, sécurité
                    physique) ou informationnel (désinformation,
                    manipulation, processus démocratique), dans le but
                    d'en réduire le risque (OUI) — ou s'agit-il d'informer
                    sur l'intervention de l'IA, de protéger contre une
                    cyberattaque, de protéger des données personnelles, ou
                    d'une régulation générale sans lien spécifique avec
                    l'IA (NON) ? Un rapprochement lexical avec les hauts
                    enjeux, la désinformation ou la sécurité ne suffit
                    jamais : ta justification doit citer l'élément précis
                    du texte qui régule l'usage à risque d'un système d'IA
                    — jamais reformuler la définition de la cible
                    elle-même. Si tu ne peux pas pointer cet élément
                    précis, la réponse est NON.
                """),
                "examples": [
                    "Un encadrement spécifique de l'utilisation de systèmes d'intelligence artificielle pour évaluer l'éligibilité à des prestations sociales.",
                    "Une interdiction d'utiliser un système d'intelligence artificielle pour prendre seul une décision de refus de crédit.",
                    "Une obligation de sécurité applicable aux robots ou machines pilotés par un système d'intelligence artificielle, destinée à prévenir le risque d'accident pour les personnes à proximité.",
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
