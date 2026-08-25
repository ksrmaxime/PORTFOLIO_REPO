# src/run_inst_prompts.py
# Un prompt système dédié par instrument (une question binaire par instrument, au lieu
# d'une seule question générique "contient-il un instrument ?" comme dans run1_prompts.py).
# La taxonomie (les 7 codes) reste définie une seule fois dans run5_prompts.py.
#
# Version simplifiée : chaque instrument est réduit à UNE question fermée, précise et
# explicite. Le LLM se perdait dans des définitions/exclusions/exemples trop longs ;
# on remplace cela par une question courte par instrument + une règle générale unique
# (pas d'extrapolation, justification d'une phrase max) appliquée à tous les prompts.
from __future__ import annotations

import pandas as pd

from src.run5_prompts import INSTRUMENT_CODES

# ---------------------------------------------------------------------------
# Une question fermée par instrument. `note` (optionnelle) précise un piège
# spécifique à cet instrument seulement ; tout le reste (interdiction
# d'extrapoler, justification courte) est géré une fois pour toutes dans
# build_system_prompt.
# ---------------------------------------------------------------------------

INSTRUMENT_DEFINITIONS: "dict[str, dict]" = {
    "VOLUNTARY": {
        "name": "instrument volontaire",
        "question": (
            "Est-ce que l'article encourage explicitement ou donne la possibilité à "
            "l'acteur régulé de s'autoréguler ?"
        ),
        "note": (
            "Une délégation de compétences au Conseil fédéral ou à un canton n'est pas "
            "considérée comme de l'autorégulation."
        ),
    },
    "TAXES_SUBSIDIES": {
        "name": "taxes et subventions",
        "question": (
            "Est-ce que l'article prévoit explicitement une subvention ou un prélèvement "
            "monétaire ?"
        ),
    },
    "PUBLIC_INVESTMENT": {
        "name": "investissement et marchés publics",
        "question": (
            "Est-ce que l'article prévoit explicitement l'investissement d'argent de "
            "l'État dans quelque chose ?"
        ),
    },
    "PROHIBITION_BAN": {
        "name": "interdiction",
        "question": "Est-ce que l'article interdit explicitement une pratique ?",
    },
    "PLANNING_EVALUATION": {
        "name": "planification et évaluation",
        "question": (
            "Est-ce que l'article prévoit explicitement une pratique permettant de "
            "tester une nouvelle politique publique ou d'en faire l'évaluation ?"
        ),
    },
    "OBLIGATION": {
        "name": "obligation",
        "question": (
            "Est-ce que l'article impose explicitement une obligation, une condition ou "
            "une contrainte ?"
        ),
    },
    "LIABILITY": {
        "name": "régime de responsabilité",
        "question": (
            "Est-ce que l'article dit explicitement si un acteur est responsable dans un "
            "certain cas de figure ?"
        ),
    },
}

assert set(INSTRUMENT_DEFINITIONS) == set(INSTRUMENT_CODES), (
    "INSTRUMENT_DEFINITIONS doit couvrir exactement les codes de run5_prompts.INSTRUMENT_CODES"
)


def build_system_prompt(code: str) -> str:
    if code not in INSTRUMENT_DEFINITIONS:
        raise KeyError(f"Unknown instrument code: {code}")

    d = INSTRUMENT_DEFINITIONS[code]
    note_block = f"\nAttention : {d['note']}\n" if d.get("note") else ""

    return (
        "Tu es un expert en analyse des politiques publiques et du droit suisse.\n\n"

        "## Question\n\n"
        f"{d['question']}\n"
        f"{note_block}\n"

        "## Règle générale\n\n"
        "- Base ta réponse UNIQUEMENT sur ce qui est écrit explicitement dans le texte. "
        "Il est interdit d'extrapoler : soit la réponse à la question est explicitement "
        "dans le texte, soit la réponse est NON.\n"
        "- Un article de loi suisse contient souvent plusieurs alinéas : il suffit qu'UN "
        "SEUL alinéa réponde explicitement OUI à la question pour que l'article entier "
        "soit classé OUI.\n"
        "- En cas de doute, réponds NON.\n\n"

        "Réponds TOUJOURS en deux parties, dans cet ordre exact, sans aucun autre texte "
        "avant, après ou entre les deux :\n"
        "Justification: [une phrase maximum, citant ou paraphrasant le passage du texte]\n"
        "Décision: OUI ou NON\n\n"
        "La ligne \"Décision:\" est OBLIGATOIRE et doit toujours être présente."
    )


USER_TEMPLATE = """Texte :
{article_text}

Réponds à la question posée dans tes instructions.

Réponds en deux parties dans cet ordre exact :
Justification: [une phrase maximum]
Décision: OUI ou NON"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)
