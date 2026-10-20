# src/ai_relevance_prompts.py
"""Step LLM 1 (v2) : le terme trouvé par le filtre par mots-clés renvoie-t-il à de l'IA ?

Trois classes (cf. PORTFOLIO_NEW_GOLD) :
  SPECIFIQUE : l'IA est nommée, ou le système visé perçoit / comprend / agit seul ;
  NEUTRE     : décision / évaluation / profilage par une machine, encadré quelle que soit la
               technique ;
  NON        : simple informatique (traitement, transmission, mécanisme fixe).

Changements par rapport à la v1 (src/OLD/ai_relevance_prompts_v1.py, job 66233220) :
  - plus de test négatif (« pourrait-il fonctionner sans IA ? ») que le modèle prenait au pied
    de la lettre ; les catégories sont décrites positivement, par ce que fait la machine ;
  - réponse en une lettre (A/B/C) : un seul token par réponse, plus de biais de longueur ;
  - quelques exemples INVENTÉS et généraux (aucun article du corpus) pour montrer le
    raisonnement, pas des réponses à recopier ;
  - les articles où l'IA est nommée explicitement (mots-clés du groupe "ia") ne passent pas
    par le LLM : SPECIFIQUE directement (cf. RULE_SPECIFIQUE_KEYWORDS).
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

LABELS: List[str] = ["SPECIFIQUE", "NEUTRE", "NON"]
LETTERS: List[str] = ["A", "B", "C"]                    # même ordre que LABELS
LETTER_TO_LABEL = dict(zip(LETTERS, LABELS))

# Libellés de mots-clés (src/keyword_filter.py, colonne kw_keywords) qui suffisent à classer
# l'article SPECIFIQUE sans LLM : l'IA y est nommée en toutes lettres.
RULE_SPECIFIQUE_KEYWORDS: List[str] = [
    "intelligence artificielle",
    "IA (sigle)",
    "système d'IA",
    "IA générative / modèle de langage",
]

SYSTEM_PROMPT = """Tu es juriste, spécialiste du droit suisse et de la régulation de l'intelligence artificielle (IA).

Un filtre par mots-clés a repéré, dans un article de loi, un terme lié à l'automatisation. Ta tâche : dire à quel type de technologie ce terme renvoie, tel qu'il est employé dans CET article.

## Les trois catégories
A — Système d'IA. Le terme désigne une machine qui accomplit elle-même une tâche demandant normalement de la perception, de la compréhension ou du jugement humains : conduire un véhicule à la place du conducteur, reconnaître une personne ou un objet sur une image, comprendre ou produire du texte, des images ou des sons, agir de façon autonome dans le monde réel. Il suffit que le texte vise un tel système, même s'il n'emploie pas les mots « intelligence artificielle ».

B — Décision ou évaluation automatisée. Le terme désigne une machine qui évalue une personne ou tranche un cas à la place d'un humain : décision rendue sans intervention humaine, profilage (évaluation automatisée de caractéristiques personnelles comme le comportement, le risque, la solvabilité ou la santé), attribution d'un score ou classement de personnes. La règle vaut quelle que soit la technique (simple logiciel ou IA). Si l'article encadre l'usage de tels résultats (autorisation, communication, droits des personnes), c'est aussi B.

C — Simple informatique. Le terme indique seulement que des données sont traitées par ordinateur, ou qu'un appareil fonctionne de lui-même selon un mécanisme fixe : enregistrer, transmettre, consulter, comparer, calculer, mesurer, déclencher. Aucune machine n'évalue de personne ni ne remplace un jugement humain. « Autonome » au sens d'« indépendant » et « traitement automatisé » au sens de « traitement informatique » relèvent de C.

## Comment raisonner
1. Regarde ce que fait la machine dans cet article : elle perçoit, comprend ou agit seule (A) ; elle évalue ou décide à propos d'une personne ou d'un cas (B) ; elle exécute seulement une opération sur des données ou un mécanisme (C).
2. Juge l'emploi du terme dans cet article précis, pas le mot en général. Le titre de l'acte et le chapitre peuvent aider.
3. Entre A et B : A si la tâche demande de percevoir ou de comprendre (voir, conduire, lire, parler) ; B si la machine applique des critères à des données sur une personne ou un cas.
4. Entre B et C : B seulement si le résultat porte sur une personne ou un cas et tient lieu d'appréciation humaine.

## Exemples (inventés, pour illustrer le raisonnement)
- « Le drone de livraison évite seul les obstacles détectés par ses caméras. » → A : la machine perçoit son environnement et agit sans pilote.
- « Les demandes peuvent être triées par un logiciel qui attribue un score de priorité à chaque requérant. » → B : une machine évalue des personnes.
- « L'office peut statuer sans qu'une personne physique examine le dossier. » → B : décision rendue sans intervention humaine.
- « Les données sont transmises par une procédure automatisée à l'autorité compétente. » → C : simple transmission informatique.
- « La vanne se ferme automatiquement lorsque le niveau dépasse le seuil fixé. » → C : mécanisme fixe, aucun jugement.

Réponds par une seule lettre : A, B ou C."""

USER_TEMPLATE = """Acte : {act_title}{act_meta}
Emplacement : {breadcrumb}
Article : {article_label}{marginal}

Texte de l'article :
{article_text}

Terme(s) repéré(s) dans cet article : {terms}

À quelle catégorie renvoie ce terme dans cet article ? Réponds par une seule lettre : A, B ou C."""

JUSTIF_SUFFIX = "\n\nJustification en une phrase, en citant le passage de l'article : "

# Entrées « vides » pour la calibration contextuelle : on mesure la préférence du modèle
# pour A/B/C sans contenu informatif, puis on la retire des probabilités.
CONTENT_FREE_INPUTS: List[Dict[str, str]] = [
    {"act_title": "N/A", "breadcrumb": "N/A", "article_label": "N/A", "article_text": "N/A", "terms": "« N/A »"},
    {"act_title": "", "breadcrumb": "", "article_label": "", "article_text": "", "terms": ""},
    {"act_title": "[ACTE]", "breadcrumb": "[EMPLACEMENT]", "article_label": "[ARTICLE]", "article_text": "[TEXTE]", "terms": "« [TERME] »"},
]


def content_free_prompts() -> List[str]:
    return [USER_TEMPLATE.format(act_meta="", marginal="", **d) for d in CONTENT_FREE_INPUTS]


def build_contexts(df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """Pour chaque ligne d'article (level 6) : titre de l'acte, fil d'Ariane (parties /
    titres / chapitres / sections) et titre marginal (heading de niveau 5 juste avant)."""
    ctx: Dict[int, Dict[str, str]] = {}
    levels = pd.to_numeric(df["level"], errors="coerce").fillna(-1).astype(int).to_numpy()
    labels = df["label"].fillna("").astype(str).to_numpy()
    laws = df["law_id"].astype(str).to_numpy()
    idx = df.index.to_numpy()

    cur_law, title = None, ""
    cur: Dict[int, str] = {}
    prev_level = None
    for i in range(len(df)):
        if laws[i] != cur_law:
            cur_law, title, cur, prev_level = laws[i], "", {}, None
        lv, lab = levels[i], labels[i]
        if lv == 0:
            title = lab
        elif 1 <= lv <= 5:
            cur[lv] = lab
            for deeper in [k for k in cur if k > lv]:
                del cur[deeper]
        elif lv == 6:
            marginal = cur.get(5, "") if prev_level == 5 else ""
            path = [cur[k] for k in sorted(cur) if k <= 4 or (k == 5 and not marginal)]
            ctx[idx[i]] = {"act_title": title, "breadcrumb": " > ".join(p for p in path if p) or "(dispositions sans subdivision)",
                           "marginal": marginal}
        prev_level = lv
    return ctx


def build_user_prompt(row: pd.Series, ctx: Dict[str, str], terms: List[str]) -> str:
    meta = []
    if str(row.get("sr_number", "") or "").strip():
        meta.append(f"RS {row['sr_number']}")
    if str(row.get("doc_type_label", "") or "").strip():
        meta.append(str(row["doc_type_label"]))
    return USER_TEMPLATE.format(
        act_title=ctx["act_title"] or "(titre inconnu)",
        act_meta=f" ({', '.join(meta)})" if meta else "",
        breadcrumb=ctx["breadcrumb"],
        article_label=str(row.get("label", "")).strip(),
        marginal=f" — {ctx['marginal']}" if ctx["marginal"] else "",
        article_text=str(row.get("text", "") or "").strip(),
        terms=", ".join(f"« {t} »" for t in terms),
    )
