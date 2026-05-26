# src/run2_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un classificateur de textes juridiques. Ta seule tâche est de "
    "déterminer si un article de loi contient un instrument qui répond à "
    "un problème public en lien avec l'intelligence artificielle.\n\n"
    "Un article est pertinent s'il adresse au moins un des problèmes "
    "publics suivants :\n\n"
    "DONNÉES\n"
    "- Protection des données personnelles : collecte, traitement, "
    "conservation ou transfert de données individuelles par des "
    "systèmes automatisés\n"
    "- Propriété intellectuelle et contenu créatif : droits d'auteur, "
    "droits voisins ou protections appliquées à des contenus générés "
    "ou traités par des systèmes automatisés\n\n"
    "COMPÉTENCES\n"
    "- Formation et éducation : développement de compétences numériques "
    "ou liées aux technologies dans les cursus ou programmes de "
    "formation\n"
    "- Recherche : financement ou organisation de la recherche "
    "scientifique sur ou avec des technologies numériques et "
    "automatisées\n\n"
    "INFRASTRUCTURE\n"
    "- Calcul et matériel informatique : accès, approvisionnement ou "
    "disponibilité stratégique de capacités de calcul ou de matériel "
    "spécialisé\n"
    "- Centres de données et énergie : construction, exploitation, "
    "sécurité ou approvisionnement énergétique des infrastructures "
    "physiques hébergeant des systèmes numériques\n\n"
    "RISQUES ET PRÉJUDICES SOCIÉTAUX\n"
    "- Gouvernance des applications à enjeux élevés : déploiement de "
    "systèmes automatisés dans des domaines sensibles comme la santé, "
    "la justice, les transports ou la finance\n"
    "- Responsabilité algorithmique : transparence, auditabilité ou "
    "contrôle humain des décisions automatisées affectant des citoyens\n"
    "- Désinformation : création, diffusion ou responsabilité liée à "
    "des contenus synthétiques ou trompeurs générés automatiquement\n"
    "- Cybersécurité des systèmes automatisés : robustesse, intégrité "
    "ou sécurité des systèmes et des données qui les alimentent\n\n"
    "RÈGLE D'INCLUSION : l'article peut être pertinent même sans mentionner "
    "explicitement l'IA, à condition que son objet premier soit de réguler "
    "des systèmes automatisés, des données traitées par machine, des "
    "compétences numériques ou une infrastructure de calcul.\n\n"
    "Ne classe PAS comme pertinent :\n"
    "- Un article dont l'objet est de réguler des comportements humains "
    "ou des procédures administratives, même si des systèmes automatisés "
    "pourraient être utilisés pour appliquer ou faire respecter ces règles : "
    "ce qui compte est ce que la loi réglemente, non les outils qui "
    "pourraient servir à l'appliquer\n"
    "- Un article qui porte sur un secteur où l'IA peut être présente "
    "mais qui ne cible pas les systèmes automatisés eux-mêmes comme "
    "objet de la régulation\n"
    "- Un article qui mentionne des données, une infrastructure ou une "
    "coordination institutionnelle sans que des systèmes automatisés "
    "en soient explicitement l'objet\n\n"
    "Réponds en deux parties, dans cet ordre exact :\n"
    "Justification: [1 à 2 phrases maximum, ancrées dans le texte]\n"
    "Décision: OUI ou NON"
)

USER_TEMPLATE = """L'article suivant contient-il un instrument qui répond à un \
problème public en lien avec l'intelligence artificielle ?

Article : {article_text}

Réponds en deux parties dans cet ordre exact :
Justification: [1 à 2 phrases maximum]
Décision: OUI ou NON"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row.get(text_col)) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
