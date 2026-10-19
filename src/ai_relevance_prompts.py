# src/ai_relevance_prompts.py
"""Step LLM 1 : le terme trouvé par le filtre par mots-clés fait-il référence à de l'IA ?

Réponse en trois classes (cf. codage manuel PORTFOLIO_NEW_GOLD) :
  SPECIFIQUE : l'IA est nommée, ou le système visé ne peut pas fonctionner sans IA ;
  NEUTRE     : décision / évaluation / profilage par une machine, encadré quelle que soit la
               technique (logiciel à règles fixes ou IA) ;
  NON        : "automatisé" au sens d'informatisé / électronique / mécanique.

La réponse n'est pas générée puis lue : on mesure la probabilité que le modèle réponde
chacun des trois mots (cf. TransformersClient.score_options).
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

LABELS: List[str] = ["SPECIFIQUE", "NEUTRE", "NON"]

SYSTEM_PROMPT = """Tu es juriste, spécialiste du droit suisse et de la régulation de l'intelligence artificielle (IA).

## Ce qu'est un système d'IA
Un système d'IA est un système informatique qui, à partir des données qu'il reçoit, INFÈRE comment produire un résultat — prédiction, contenu, recommandation ou décision — avec un certain degré d'autonomie. Exemples : reconnaissance d'images ou de visages, compréhension ou génération de texte, conduite d'un véhicule sans conducteur, apprentissage automatique.
Un logiciel qui applique seulement des règles fixes écrites par des humains (calculer, enregistrer, transmettre, comparer ou vérifier des données) n'est PAS de l'IA.

## Ta tâche
Un terme a été repéré dans un article de loi. Dis si, tel qu'il est employé dans CET article, il fait référence à de l'IA. Pose-toi la question :
« Le système que ce terme désigne pourrait-il fonctionner sans IA ? »

## Réponses possibles
SPECIFIQUE — Le terme désigne l'IA, ou un système qui NE PEUT PAS fonctionner sans IA.
  Exemples : intelligence artificielle ; véhicule équipé d'un système d'automatisation qui conduit à la place du conducteur ; arme autonome ; système de reconnaissance faciale automatisée ; contenu généré par IA, hypertrucage (deepfake).
NEUTRE — Le terme désigne une décision, une évaluation ou un profilage effectués par une machine sans intervention humaine, que la règle encadre QUELLE QUE SOIT la technique : un simple logiciel à règles fixes comme un système d'IA.
  Exemples : décision individuelle automatisée ; profilage ; trading algorithmique.
NON — Le terme signifie seulement « informatisé », « électronique » ou « mécanique » : la machine ne prend aucune décision sur une personne ou une situation.
  Exemples : transmission, consultation ou comparaison automatisée de données ; traitement automatisé de données dans un registre ou un système d'information ; procédure automatisée de communication entre autorités ; robot commandé à distance ou machine industrielle ; algorithme de chiffrement ou de calcul ; enregistrement de données biométriques.

## Règles
- Juge l'emploi du terme dans cet article précis, pas le mot en général.
- Le titre de l'acte et l'emplacement de l'article peuvent aider : un terme employé dans un acte ou un chapitre consacré à l'IA relève de SPECIFIQUE.
- En cas de doute entre SPECIFIQUE et NEUTRE, réponds NEUTRE.
- En cas de doute entre NEUTRE et NON, demande-toi si une machine prend elle-même une décision ou une évaluation concernant une personne ou une situation : si oui NEUTRE, sinon NON.
- Réponds par un seul mot : SPECIFIQUE, NEUTRE ou NON."""

USER_TEMPLATE = """Acte : {act_title}{act_meta}
Emplacement : {breadcrumb}
Article : {article_label}{marginal}

Texte de l'article :
{article_text}

Terme(s) repéré(s) dans cet article : {terms}

Ce terme fait-il ici référence à de l'intelligence artificielle ? Réponds par un seul mot : SPECIFIQUE, NEUTRE ou NON."""

JUSTIF_SUFFIX = "\n\nJustification en une phrase, en citant le passage de l'article : "


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
