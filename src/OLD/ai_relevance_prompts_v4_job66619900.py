# src/ai_relevance_prompts.py
"""Step LLM 1 (v4 : SPECIFIQUE inclut le cas 3 « risque ou capacité dont l'IA est le moteur principal ») : le terme trouvé par le filtre par mots-clés renvoie-t-il à de l'IA ?

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

Changements v3 (après le job 66278177) :
  - la définition du terme donnée ailleurs dans le même acte (article « Définitions »,
    « on entend par… ») est ajoutée au prompt : beaucoup d'articles parlent d'un « système
    d'automatisation » sans dire ce qu'il fait, c'est la définition qui le dit ;
  - règle de raisonnement : c'est le système désigné qui compte, pas l'opération que l'article
    décrit (enregistrer ses données, l'entretenir, l'autoriser…) ;
  - propagation intra-acte (cf. PROPAGATION_KEYWORDS) appliquée après le LLM dans le script ;
  - probabilités brutes par défaut (la calibration dégradait les résultats).
"""
from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

LABELS: List[str] = ["SPECIFIQUE", "NEUTRE", "NON"]
LETTERS: List[str] = ["A", "B", "C"]                    # même ordre que LABELS
LETTER_TO_LABEL = dict(zip(LETTERS, LABELS))

# Libellés de mots-clés (src/keyword_filter.py, colonne kw_keywords) qui suffisent à classer
# l'article SPECIFIQUE sans LLM : l'IA y est nommée en toutes lettres.
# Propagation intra-acte : si un article d'un acte est SPECIFIQUE, les autres articles du même
# acte qui contiennent un de ces mots-clés (également présent dans l'article SPECIFIQUE)
# deviennent SPECIFIQUE. Un terme défini dans un acte garde le même sens dans tout l'acte.
# Seuls les termes qui désignent un SYSTÈME sont propagés (pas « traitement automatisé »,
# « profilage », « algorithme »…, dont le sens varie d'un article à l'autre).
PROPAGATION_KEYWORDS: List[str] = [
    "conduite automatisée / véhicule automatisé",
    "armes autonomes",
    "système d'automatisation / système autonome",
    "reconnaissance faciale / biométrique",
    "hypertrucage / deepfake",
    "chatbot / agent conversationnel",
    "apprentissage automatique / machine learning",
    "apprentissage profond / réseau de neurones",
]

RULE_SPECIFIQUE_KEYWORDS: List[str] = [
    "intelligence artificielle",
    "IA (sigle : l'IA, d'IA)",
    "système d'IA",
    "IA générative / modèle de langage",
]

SYSTEM_PROMPT = """Tu es juriste, spécialiste du droit suisse et de la régulation de l'intelligence artificielle (IA).

Un filtre par mots-clés a repéré, dans un article de loi, un terme lié à l'automatisation ou aux technologies numériques. Ta tâche : dire à quel type de technologie ce terme renvoie, tel qu'il est employé dans CET article.

## Les trois catégories
A — IA. Le terme renvoie à l'IA dans l'un de ces trois cas :
1. un système d'IA : une machine qui accomplit elle-même une tâche demandant normalement de la perception, de la compréhension ou du jugement humains (conduire un véhicule à la place du conducteur, reconnaître une personne ou un objet sur une image, comprendre ou produire du texte, des images ou des sons, agir de façon autonome dans le monde réel) ;
2. un système qui ne peut pas fonctionner sans IA ;
3. un risque ou une capacité dont l'IA est aujourd'hui le moteur principal : manipulation de l'information et de l'opinion, atteinte à la capacité de penser de façon autonome (intégrité ou espace cognitif), contenus synthétiques ou truqués, et les « technologies émergentes » lorsqu'elles sont évoquées à propos de ces risques.
Il suffit que le texte vise l'un de ces cas, même s'il n'emploie pas les mots « intelligence artificielle ».

B — Décision ou évaluation automatisée. Le terme désigne une machine qui évalue une personne ou tranche un cas à la place d'un humain : décision rendue sans intervention humaine, profilage (évaluation automatisée de caractéristiques personnelles comme le comportement, le risque, la solvabilité ou la santé), attribution d'un score ou classement de personnes. La règle vaut quelle que soit la technique (simple logiciel ou IA). Si l'article encadre l'usage de tels résultats (autorisation, communication, droits des personnes), c'est aussi B.

C — Simple informatique ou numérisation. Le terme indique seulement que des données sont traitées par ordinateur, ou qu'un appareil fonctionne de lui-même selon un mécanisme fixe : enregistrer, transmettre, consulter, comparer, calculer, mesurer, déclencher. Aucune machine n'évalue de personne ni ne remplace un jugement humain. Relèvent aussi de C : « autonome » au sens d'« indépendant », « traitement automatisé » au sens de « traitement informatique », la numérisation en général (transformation numérique, cloud, systèmes d'information), la cybersécurité classique et les engins télécommandés, tant que l'article ne vise pas l'un des trois cas de A.

## Comment raisonner
1. Regarde ce que fait la machine dans cet article : elle perçoit, comprend ou agit seule (A) ; elle évalue ou décide à propos d'une personne ou d'un cas (B) ; elle exécute seulement une opération sur des données ou un mécanisme (C).
2. Juge l'emploi du terme dans cet article précis, pas le mot en général. Le titre de l'acte et le chapitre peuvent aider.
3. Entre A et B : A si la tâche demande de percevoir ou de comprendre (voir, conduire, lire, parler) ; B si la machine applique des critères à des données sur une personne ou un cas.
4. Entre B et C : B seulement si le résultat porte sur une personne ou un cas et tient lieu d'appréciation humaine.
5. C'est le système désigné par le terme qui compte, pas l'opération que l'article décrit. Un article peut ne parler que des données d'un système, de son entretien, de son autorisation ou de son contrôle : si le système lui-même perçoit, reconnaît ou conduit, la réponse est A. Si une définition du terme est fournie, appuie-toi dessus.
6. Pour un risque (cas 3 de A) : la réponse est A quand l'article vise la manipulation de l'information ou de l'opinion, l'autonomie de pensée des personnes ou les contenus truqués, même formulés en termes généraux (« technologies émergentes », « espace cognitif »).

## Exemples (inventés, pour illustrer le raisonnement)
- « Le drone de livraison évite seul les obstacles détectés par ses caméras. » → A : la machine perçoit son environnement et agit sans pilote.
- « Les demandes peuvent être triées par un logiciel qui attribue un score de priorité à chaque requérant. » → B : une machine évalue des personnes.
- « L'office peut statuer sans qu'une personne physique examine le dossier. » → B : décision rendue sans intervention humaine.
- « Les données sont transmises par une procédure automatisée à l'autorité compétente. » → C : simple transmission informatique.
- « La vanne se ferme automatiquement lorsque le niveau dépasse le seuil fixé. » → C : mécanisme fixe, aucun jugement.
- « La Confédération renforce la résilience de la population face à la manipulation de l'opinion par les technologies émergentes. » → A : risque dont l'IA est le moteur principal (cas 3).
- « Un crédit est alloué à la transformation numérique de l'administration. » → C : numérisation en général.

Réponds par une seule lettre : A, B ou C."""

USER_TEMPLATE = """Acte : {act_title}{act_meta}
Emplacement : {breadcrumb}
{definitions}Article : {article_label}{marginal}

Texte de l'article :
{article_text}

Terme(s) repéré(s) dans cet article : {terms}

À quelle catégorie renvoie ce terme dans cet article ? Réponds par une seule lettre : A, B ou C."""

JUSTIF_SUFFIX = "\n\nJustification en une phrase, en tes propres mots, en citant le passage de l'article qui montre ce que fait la machine : "

DEFINITIONS_HEADER = "Définition dans le même acte ({label}) :\n{text}\n\n"
MAX_DEF_CHARS = 900

# Entrées « vides » pour la calibration contextuelle : on mesure la préférence du modèle
# pour A/B/C sans contenu informatif, puis on la retire des probabilités.
CONTENT_FREE_INPUTS: List[Dict[str, str]] = [
    {"act_title": "N/A", "breadcrumb": "N/A", "article_label": "N/A", "article_text": "N/A", "terms": "« N/A »"},
    {"act_title": "", "breadcrumb": "", "article_label": "", "article_text": "", "terms": ""},
    {"act_title": "[ACTE]", "breadcrumb": "[EMPLACEMENT]", "article_label": "[ARTICLE]", "article_text": "[TEXTE]", "terms": "« [TERME] »"},
]


def content_free_prompts() -> List[str]:
    return [USER_TEMPLATE.format(act_meta="", marginal="", definitions="", **d) for d in CONTENT_FREE_INPUTS]


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


_DEF_RX = re.compile(r"on entend par|au sens de la pr[ée]sente|au sens du pr[ée]sent|sont r[ée]put[ée]s", re.I)
_DEF_MARGINAL_RX = re.compile(r"d[ée]finition|notion|terminologie", re.I)


def find_definition_articles(df: pd.DataFrame) -> Dict[str, List[int]]:
    """law_id -> index des articles (level 6) qui donnent des définitions : titre marginal
    « Définition(s) / Notion / Terminologie » ou formule « on entend par », « au sens de la
    présente… »."""
    out: Dict[str, List[int]] = {}
    levels = pd.to_numeric(df["level"], errors="coerce").fillna(-1).astype(int).to_numpy()
    labels = df["label"].fillna("").astype(str).to_numpy()
    texts = df["text"].fillna("").astype(str).to_numpy()
    laws = df["law_id"].astype(str).to_numpy()
    idx = df.index.to_numpy()
    for i in range(len(df)):
        if levels[i] != 6:
            continue
        marginal = labels[i - 1] if i > 0 and levels[i - 1] == 5 and laws[i - 1] == laws[i] else ""
        if _DEF_MARGINAL_RX.search(marginal) or _DEF_RX.search(texts[i][:3000]):
            out.setdefault(laws[i], []).append(idx[i])
    return out


def definitions_for(df: pd.DataFrame, i: int, def_index: Dict[str, List[int]], keywords, find_in_text) -> str:
    """Passages des articles de définitions du même acte qui contiennent un des mots-clés
    trouvés dans l'article i (seulement les lignes / lettres concernées). Chaîne vide sinon."""
    law = str(df.at[i, "law_id"])
    blocks = []
    for j in def_index.get(law, []):
        if j == i:
            continue
        text = str(df.at[j, "text"] or "")
        if not find_in_text(text, keywords):
            continue
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        keep = text.strip() if len(text) <= MAX_DEF_CHARS else "\n".join(ln for ln in lines if find_in_text(ln, keywords))
        if keep:
            blocks.append(DEFINITIONS_HEADER.format(label=str(df.at[j, "label"]).strip(), text=keep[:MAX_DEF_CHARS]))
    return "".join(blocks[:2])


def build_user_prompt(row: pd.Series, ctx: Dict[str, str], terms: List[str], definitions: str = "") -> str:
    meta = []
    if str(row.get("sr_number", "") or "").strip():
        meta.append(f"RS {row['sr_number']}")
    if str(row.get("doc_type_label", "") or "").strip():
        meta.append(str(row["doc_type_label"]))
    return USER_TEMPLATE.format(
        act_title=ctx["act_title"] or "(titre inconnu)",
        act_meta=f" ({', '.join(meta)})" if meta else "",
        breadcrumb=ctx["breadcrumb"],
        definitions=("\n" + definitions) if definitions else "",
        article_label=str(row.get("label", "")).strip(),
        marginal=f" — {ctx['marginal']}" if ctx["marginal"] else "",
        article_text=str(row.get("text", "") or "").strip(),
        terms=", ".join(f"« {t} »" for t in terms),
    )
