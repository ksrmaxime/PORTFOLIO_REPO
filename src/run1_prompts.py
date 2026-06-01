# src/run1_prompts.py (portfolio run1)
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un expert en analyse des politiques publiques et du droit suisse.\n\n"
    "Ta tâche est de déterminer si un article de loi contient un INSTRUMENT DE POLITIQUE PUBLIQUE.\n\n"
    "Un instrument de politique publique est toute disposition par laquelle l'État impose, interdit, incite, "
    "finance, autorise, surveille ou sanctionne un comportement d'acteurs (individus, entreprises, "
    "organisations, autorités).\n\n"
    "Les catégories d'instruments recherchés sont :\n"
    "1. **Voluntary instruments** : actions de l'État pour promouvoir un comportement sans contrainte directe — "
    "campagnes de sensibilisation, encouragements (\"encourage\", \"peut coordonner\"), programmes de "
    "prévention, codes de conduite, accords volontaires\n"
    "2. **Taxes and subsidies** : taxes, impôts, redevances, subventions, allocations, aides financières, "
    "exonérations fiscales\n"
    "3. **Public investment & public procurement** : investissements publics, marchés publics, achats de l'État\n"
    "4. **Prohibition & Ban** : interdictions, bans, restrictions d'activités ou de comportements, conditions "
    "restrictives (\"ne peut...que si\", \"est interdit sauf\")\n"
    "5. **Planning & evaluation instruments** : plans, programmes, évaluations obligatoires, rapports, "
    "registres, inventaires, gestion et évaluation des risques\n"
    "6. **Obligation** : obligations légales de faire ou de ne pas faire imposées à des acteurs privés ou "
    "publics — exigences réglementaires, normes, marquages obligatoires, demandes d'autorisation, licences, "
    "obligations d'information ou de notification, délais de mise en conformité\n"
    "7. **Liability schemes** : responsabilité civile, saisie et confiscation, assurance obligatoire, "
    "indemnisation, mesures d'exécution et sanctions administratives\n\n"
    "Points d'attention — les situations suivantes CONTIENNENT un instrument malgré les apparences :\n"
    "- Un article attribuant à une autorité la compétence d'INTERDIRE, RESTREINDRE ou OBLIGER des acteurs "
    "= instrument (la compétence d'exercer un instrument EST elle-même un instrument)\n"
    "- Une obligation imposée à un acteur PRIVÉ dans un contexte procédural "
    "= instrument (ex. : obligation de marquage, d'information, de demande d'autorisation)\n"
    "- Un article autorisant des saisies, confiscations ou mesures d'exécution "
    "= instrument (Liability scheme)\n"
    "- Une disposition transitoire imposant une obligation avec délai (dépôt de demande, mise en conformité) "
    "= instrument\n"
    "- Des obligations de sécurité ou d'évaluation des risques imposées à des organisations "
    "= instrument (Obligation + Planning)\n\n"
    "Un article NE contient PAS d'instrument uniquement si c'est :\n"
    "- Une pure définition terminologique sans obligation ni interdiction\n"
    "- Un pur renvoi à d'autres articles ou lois sans exigence propre\n"
    "- Un champ d'application sans aucune exigence comportementale\n"
    "- Une disposition interne entre autorités sans effet sur des acteurs extérieurs\n\n"
    "En cas de doute, réponds OUI.\n\n"
    "Réponds UNIQUEMENT par OUI ou NON (sans explication).\n"
    "- OUI = l'article contient au moins un instrument de politique publique\n"
    "- NON = l'article ne contient pas d'instrument de politique publique"
)

USER_TEMPLATE = """Texte :
{article_text}

Cet article contient-il un instrument de politique publique ?"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
