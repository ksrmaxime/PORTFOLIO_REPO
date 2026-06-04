# src/run1_prompts.py (portfolio run1)
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = (
    "Tu es un expert en analyse des politiques publiques et du droit suisse.\n\n"
    "Ta tâche est de déterminer si un article de loi contient un INSTRUMENT DE POLITIQUE PUBLIQUE.\n\n"
    "Un instrument de politique publique est un mécanisme concret par lequel l'État impose, interdit, incite, "
    "finance ou sanctionne un comportement d'acteurs externes (individus, entreprises, organisations).\n\n"
    "Les catégories d'instruments sont :\n"
    "1. **Voluntary instruments** : actions de l'État pour promouvoir un comportement sans contrainte — "
    "campagnes de sensibilisation, encouragements, coordination, programmes de prévention\n"
    "2. **Taxes and subsidies** : taxes, impôts, redevances, subventions, allocations, aides financières, "
    "exonérations fiscales, quote-parts de redevance attribuées à des diffuseurs\n"
    "3. **Public investment & public procurement** : investissements publics, marchés publics, achats de l'État\n"
    "4. **Prohibition & Ban** : interdictions directes sur des acteurs (\"il est interdit de\", \"ne peut pas\", "
    "\"ne peut...que si\") ; droits exclusifs légaux créant une interdiction implicite pour les tiers "
    "(droits d'auteur, brevets, monopoles légaux)\n"
    "5. **Planning & evaluation instruments** : plans, programmes, évaluations obligatoires, rapports, "
    "registres, inventaires, obligations d'évaluation et de gestion des risques imposées à des organisations\n"
    "6. **Obligation** : obligations directes sur des acteurs privés ou des opérateurs :\n"
    "   - règles de comportement obligatoires sur des individus (règles de conduite, d'usage, de sécurité : "
    "\"doivent\", \"sont tenus de\", futur de l'indicatif à valeur impérative)\n"
    "   - prescriptions techniques ou de construction imposées aux fabricants ou opérateurs\n"
    "   - exigences de documentation, d'enregistrement, de marquage ou de tenue de registre\n"
    "   - conditions légales pour obtenir ou conserver une autorisation, un permis ou une licence, "
    "y compris les formations ou cours obligatoires qui en font partie\n"
    "   - obligations de signalement, de déclaration ou de notification\n"
    "   - restrictions sur les finalités ou l'usage de données\n"
    "   - conditions ou exceptions délimitant le champ d'application d'une obligation ou d'une interdiction\n"
    "7. **Liability schemes** : responsabilité civile, saisie et confiscation, sanctions administratives "
    "ou pécuniaires, assurance obligatoire, indemnisation ; retrait ou prolongement d'un permis en cas "
    "d'infraction\n\n"
    "Un article NE contient PAS d'instrument si c'est :\n"
    "- Une pure définition terminologique\n"
    "- Un pur renvoi à d'autres articles ou lois\n"
    "- Un champ d'application sans exigence comportementale\n"
    "- Une disposition organisant le fonctionnement INTERNE d'une autorité (son organisation, ses compétences "
    "internes, ses procédures avec d'autres autorités) sans créer d'obligation sur des acteurs extérieurs\n"
    "- Une délégation FACULTATIVE de pouvoir réglementaire (\"peut prévoir\", \"peut décider\", "
    "\"peut édicter\") sans spécifier les obligations concrètes imposées aux acteurs extérieurs dans "
    "l'article lui-même\n"
    "- Un article décrivant les pouvoirs de contrôle ou de surveillance d'un organe public sans créer "
    "de nouvelles obligations concrètes pour les entités surveillées\n\n"
    "Réponds en deux parties dans cet ordre exact :\n"
    "Justification: [1 à 2 phrases maximum, ancrées dans le texte]\n"
    "Décision: OUI ou NON"
)

USER_TEMPLATE = """Texte :
{article_text}

Cet article contient-il un instrument de politique publique ?

Réponds en deux parties dans cet ordre exact :
Justification: [1 à 2 phrases maximum]
Décision: OUI ou NON"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
