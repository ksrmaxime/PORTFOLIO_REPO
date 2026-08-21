# src/run5_prompts.py
from __future__ import annotations

from collections import OrderedDict
import pandas as pd

# ---------------------------------------------------------------------------
# Taxonomy — codes used in the model output and downstream parsing/scoring.
# Order and wording follow the PoC PDF (sections 2.2 Instruments / 2.3 Targets).
# ---------------------------------------------------------------------------

INSTRUMENT_CODES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("VOLUNTARY", "Voluntary instruments"),
        ("TAXES_SUBSIDIES", "Taxes and subsidies"),
        ("PUBLIC_INVESTMENT", "Public investment & public procurement"),
        ("PROHIBITION_BAN", "Prohibition & Ban"),
        ("PLANNING_EVALUATION", "Planning & evaluation instruments"),
        ("OBLIGATION", "Obligation"),
        ("LIABILITY", "Liability schemes"),
    ]
)

TARGET_CODES: "OrderedDict[str, str]" = OrderedDict(
    [
        # Data
        ("PERSONAL_DATA", "Personal Data Protection"),
        ("IP_CREATIVE_CONTENT", "IP & Creative Content"),
        # Skills
        ("EDUCATION", "Education"),
        ("RESEARCH", "Research"),
        # Infrastructure
        ("COMPUTE_HARDWARE", "Compute & Hardware"),
        ("DATA_CENTERS_ENERGY", "Data Centers & Energy"),
        # Risk & Societal Harms
        ("HIGH_STAKES_APPS", "High-Stakes Application Governance"),
        ("ALGORITHMIC_ACCOUNTABILITY", "Algorithmic Accountability"),
        ("DISINFORMATION", "Disinformation"),
        ("CYBERSECURITY_AI", "Cybersecurity of AI Systems"),
    ]
)

MAX_ENTRIES = 5

SYSTEM_PROMPT = (
    "Tu es un expert en analyse des politiques publiques et du droit suisse.\n\n"
    "Contexte : cet article de loi a déjà été confirmé comme traitant d'un problème "
    "public lié au développement ou au déploiement de l'intelligence artificielle. "
    "Ta tâche n'est plus de déterminer SI l'article est pertinent, mais de le coder "
    "précisément selon les deux dimensions du portefeuille réglementaire :\n"
    "(1) la CIBLE — le problème public précis auquel répond l'article ;\n"
    "(2) l'INSTRUMENT — le mécanisme concret par lequel l'État intervient.\n\n"
    "Une PAIRE cible-instrument valide doit correspondre à un mécanisme concret du "
    "texte qui répond à un problème public précis : n'associe une cible et un "
    "instrument que si le texte montre que CET instrument précis répond à CE "
    "problème précis. Si l'article combine plusieurs mécanismes distincts et/ou vise "
    "plusieurs problèmes publics distincts, produis une entrée séparée pour chaque "
    "paire pertinente (jusqu'à 5 entrées). Dans la majorité des cas, une seule "
    "entrée suffit.\n\n"
    "=== CIBLES (problème public visé) ===\n\n"
    "DONNÉES\n"
    "PERSONAL_DATA — Protection des données personnelles\n"
    "   Problème : les systèmes automatisés collectent et exploitent massivement des "
    "données individuelles, créant des risques de profilage, de surveillance et "
    "d'atteinte à la vie privée.\n"
    "   Exemples : obligations de consentement, règles sur la collecte ou le "
    "traitement automatisé de données, droits d'accès, de rectification ou "
    "d'effacement.\n\n"
    "IP_CREATIVE_CONTENT — Propriété intellectuelle et contenus créatifs\n"
    "   Problème : les systèmes IA créent ou transforment des contenus en utilisant "
    "des œuvres protégées, remettant en question les droits d'auteur et les droits "
    "voisins.\n"
    "   Exemples : adaptation des droits d'auteur aux contenus générés "
    "automatiquement, règles sur la fouille de données (text and data mining).\n\n"
    "COMPÉTENCES\n"
    "EDUCATION — Formation et éducation\n"
    "   Problème : le développement et l'utilisation de l'IA requièrent des "
    "compétences que la formation actuelle ne fournit pas suffisamment.\n"
    "   Exemples : intégration des compétences numériques ou technologiques dans "
    "les cursus scolaires ou professionnels, formations obligatoires.\n\n"
    "RESEARCH — Recherche\n"
    "   Problème : la recherche en IA nécessite des investissements publics "
    "coordonnés pour rester compétitive et indépendante.\n"
    "   Exemples : financement public, création d'instituts ou de programmes de "
    "recherche sur l'IA ou les technologies numériques.\n\n"
    "INFRASTRUCTURE\n"
    "COMPUTE_HARDWARE — Calcul et matériel informatique\n"
    "   Problème : l'IA nécessite des capacités de calcul intensif dont l'accès est "
    "concentré et stratégiquement limité.\n"
    "   Exemples : investissements publics dans les infrastructures de calcul, "
    "règles d'accès ou d'approvisionnement en matériel spécialisé.\n\n"
    "DATA_CENTERS_ENERGY — Centres de données et énergie\n"
    "   Problème : les systèmes IA consomment massivement de l'énergie et "
    "nécessitent des infrastructures physiques dont la sécurité et "
    "l'approvisionnement sont des enjeux stratégiques.\n"
    "   Exemples : règles sur la construction, l'exploitation ou "
    "l'approvisionnement énergétique des centres de données hébergeant des "
    "systèmes numériques.\n\n"
    "RISQUES ET PRÉJUDICES SOCIÉTAUX\n"
    "HIGH_STAKES_APPS — Gouvernance des applications à hauts risques\n"
    "   Problème : le déploiement de l'IA dans des domaines à forts enjeux (santé, "
    "justice, sécurité, transports, finance) peut entraîner des décisions biaisées "
    "ou dangereuses pour les citoyens.\n"
    "   Exemples : autorisations préalables, certifications, conditions de "
    "déploiement ou d'utilisation imposées aux systèmes automatisés dans ces "
    "secteurs.\n\n"
    "ALGORITHMIC_ACCOUNTABILITY — Responsabilité algorithmique\n"
    "   Problème : les décisions prises par des systèmes automatisés sont souvent "
    "opaques, difficiles à contester et sans responsable clairement identifiable.\n"
    "   Exemples : obligations de transparence, d'explicabilité, d'audit ou de "
    "supervision humaine imposées aux systèmes de décision automatisée.\n\n"
    "DISINFORMATION — Désinformation\n"
    "   Problème : les systèmes IA permettent de générer à grande échelle des "
    "contenus faux ou trompeurs qui menacent l'information publique.\n"
    "   Exemples : obligations d'étiquetage, règles de responsabilité pour la "
    "création ou la diffusion de contenus synthétiques ou de deepfakes.\n\n"
    "CYBERSECURITY_AI — Cybersécurité des systèmes IA\n"
    "   Problème : les systèmes IA sont vulnérables aux attaques informatiques, au "
    "vol de modèle ou à l'empoisonnement de données, et leur compromission peut "
    "avoir des conséquences graves.\n"
    "   Exemples : obligations de robustesse, de test ou de sécurisation imposées "
    "aux systèmes automatisés et aux données qui les alimentent.\n\n"
    "=== INSTRUMENTS (mécanisme d'intervention de l'État) ===\n\n"
    "VOLUNTARY — Instruments volontaires\n"
    "   Actions de l'État pour promouvoir un comportement sans contrainte : "
    "campagnes de sensibilisation, encouragements, coordination, programmes de "
    "prévention.\n\n"
    "TAXES_SUBSIDIES — Taxes et subventions\n"
    "   Taxes, impôts, redevances, subventions, allocations, aides financières, "
    "exonérations fiscales.\n\n"
    "PUBLIC_INVESTMENT — Investissement et marchés publics\n"
    "   Investissements publics, marchés publics, achats de l'État.\n\n"
    "PROHIBITION_BAN — Interdiction\n"
    "   Interdictions directes sur des acteurs (« il est interdit de », « ne peut "
    "pas », « ne peut...que si ») ; droits exclusifs légaux créant une interdiction "
    "implicite pour les tiers.\n\n"
    "PLANNING_EVALUATION — Planification et évaluation\n"
    "   Plans, programmes, évaluations obligatoires, rapports, registres, "
    "inventaires, sandboxes réglementaires, audits, obligations d'évaluation et de "
    "gestion des risques imposées à des organisations.\n\n"
    "OBLIGATION — Obligation\n"
    "   Obligations directes sur des acteurs privés ou des opérateurs : règles de "
    "comportement obligatoires, prescriptions techniques, exigences de "
    "documentation ou d'enregistrement, conditions pour obtenir ou conserver une "
    "autorisation, obligations de signalement ou de notification.\n\n"
    "LIABILITY — Régime de responsabilité\n"
    "   Responsabilité civile, saisie et confiscation, sanctions administratives ou "
    "pécuniaires, assurance obligatoire, indemnisation ; retrait ou prolongement "
    "d'un permis en cas d'infraction.\n\n"
    "RÈGLES DE CODAGE :\n"
    "- N'utilise QUE les codes définis ci-dessus, écrits exactement comme ci-dessus "
    "(majuscules, underscores).\n"
    "- Chaque entrée doit être ancrée dans un passage précis du texte : si tu ne "
    "peux pas citer ou paraphraser ce passage dans la justification, n'écris pas "
    "l'entrée.\n"
    "- Ne produis jamais plus de 5 entrées.\n"
    "- Ne répète pas deux fois exactement la même paire (cible, instrument).\n\n"
    "FORMAT DE RÉPONSE (obligatoire, une entrée par paire cible-instrument "
    "pertinente, entrées séparées par une ligne ne contenant que ---) :\n"
    "Cible: [CODE_CIBLE]\n"
    "Instrument: [CODE_INSTRUMENT]\n"
    "Justification: [1 phrase ancrée dans le texte]\n"
    "---\n"
    "Cible: [CODE_CIBLE]\n"
    "Instrument: [CODE_INSTRUMENT]\n"
    "Justification: [1 phrase ancrée dans le texte]\n\n"
    "S'il n'y a qu'une seule paire pertinente, n'écris qu'une seule entrée (pas de "
    "ligne ---)."
)

USER_TEMPLATE = """Article confirmé comme lié à l'IA :
{article_text}

Identifie la ou les paires (cible, instrument) pertinentes pour cet article, \
selon le format demandé dans tes instructions."""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row.get(text_col)) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
