from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from portfolio_repo.llm.client import LocalLLMClient


@dataclass(frozen=True)
class Run2Config:
    # LLM
    temperature: float = 0.8
    max_tokens: int = 900

    # output columns
    out_selected_col: str = "run2_selected"
    out_ai_relevant_col: str = "article_ai_relevant"
    out_targets_col: str = "article_targets"           # JSON string list
    out_justification_col: str = "article_justification"

    # if True, skip already-classified articles (resume behavior)
    skip_if_already_done: bool = True


_SYSTEM_PROMPT = (
    "Rôle: codeur juridique. Tu appliques une règle de codage, pas une opinion sur 'ce qui est de l'IA'.\n\n"

    "Définition opérationnelle (domain-agnostic) d'un 'système automatisé/algorithmique':\n"
    "- tout système, logiciel, procédé ou dispositif qui exécute des fonctions de manière automatisée (ex: 'automatisation', "
    "'traitement automatisé', 'décision automatisée', 'profilage', 'algorithme', 'système informatique', 'logiciel', "
    "'modèle', 'apprentissage', 'enregistreur/journal', 'interface', 'liaison de communication').\n\n"

    "Règle d'interprétation obligatoire:\n"
    "- Si un article régule un objet 'X équipé d'un système Y' (ou 'X utilisant Y'), alors l'article régule aussi le système Y "
    "dès qu'il fixe des conditions d'utilisation/admission/surveillance/obligations/autorisation/essais relatives à X+Y.\n\n"

    "Mapping targets (à appliquer):\n"
    "- Development & Adoption: autorise/interdit/conditionne l'usage, le déploiement, l'admission, les essais, la supervision, "
    "les rôles/responsabilités liés à un système automatisé.\n"
    "- Data: encadre données destinées à être traitées/produites par un système automatisé (collecte, journalisation/logs, accès, "
    "partage, effacement, exactitude/intégrité, réutilisation).\n"
    "- Computing Infrastructure: exigences techniques de sécurité/accès/stockage/interface/communication (accès non autorisé, "
    "protection, intégrité, disponibilité).\n"
    "- Skills: formation/compétences/qualifications explicitement numériques/IA.\n\n"

    "Décision:\n"
    "- Si un système automatisé est présent ET l'article fixe des conditions/règles sur son usage/admission/surveillance/etc. "
    "=> ai_relevant=true et targets inclut AU MOINS 'Development & Adoption'.\n"
    "- Sinon => ai_relevant=false.\n\n"

    "Sortie: UNIQUEMENT du JSON valide."
)




_USER_PROMPT_TEMPLATE = """Analyse UN article. Retour JSON strict uniquement.

IDs:
- law_id: {law_id}
- node_id: {node_id}
- row_uid: {row_uid}

Texte:
\"\"\"{article_text}\"\"\"

Tâche (obligatoire, dans cet ordre):
1) EXTRACTION: liste 1-5 'objets régulés' (mots du texte) + 1-5 'actions normatives' (autoriser, fixer, définir, exiger, interdire, surveiller, enregistrer, protéger, traiter, effacer, transmettre, etc.).
2) DÉTECTION: est-ce qu'un système automatisé/algorithmique est présent (true/false) ? Si true, cite le terme déclencheur exact (string court).
3) CLASSIFICATION: ai_relevant + targets + justification (1-3 phrases).

Rappel: si le texte parle d'un objet 'équipé d'un système Y' et régule des conditions d'usage/admission/essais/surveillance, alors Y est régulé.

Format EXACT:
{{
  "row_uid": {row_uid},
  "regulated_objects": [],
  "normative_actions": [],
  "automation_present": false,
  "automation_trigger": "",
  "ai_relevant": false,
  "targets": [],
  "justification": ""
}}
"""






_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_BOOL_FIXES = [
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
]


def _try_repair_json(text: str) -> str:
    t = text.strip()
    m = _JSON_BLOCK_RE.search(t)
    if m:
        t = m.group(0).strip()
    for rx, repl in _BOOL_FIXES:
        t = rx.sub(repl, t)
    t = _TRAILING_COMMA_RE.sub(r"\1", t)
    if '"' not in t and "'" in t:
        t = t.replace("'", '"')
    return t


def _safe_json_loads(raw: str) -> dict:
    return json.loads(_try_repair_json(raw))


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["law_id", "order_index", "level", "node_id"] if c in df.columns]
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def compute_run2_selection(
    df: pd.DataFrame,
    title_relevant_col: str = "title_ai_relevant",
    out_selected_col: str = "run2_selected",
) -> pd.DataFrame:
    """
    Mark articles (level=5) that are under an "active" relevant title (levels 0..4).

    Logic (per law, ordered):
    - maintain a stack for levels 0..4 of booleans (title_ai_relevant)
    - when we see a non-article (level<5), we set stack[level]=bool(value) and clear deeper levels
    - an article is selected if ANY ancestor in stack is True
    """
    if title_relevant_col not in df.columns:
        raise ValueError(f"Missing column {title_relevant_col}. Run 1 output required.")

    df2 = _stable_sort(df).copy()
    df2[out_selected_col] = pd.NA

    for law_id, grp_idx in df2.groupby("law_id", sort=False).groups.items():
        idx_list = list(grp_idx)
        active: List[Optional[bool]] = [None, None, None, None, None]  # levels 0..4

        for i in idx_list:
            lvl = int(df2.at[i, "level"])
            if lvl < 5:
                val = df2.at[i, title_relevant_col]
                # val should be True/False for lvl!=5
                b = bool(val) if pd.notna(val) else False
                active[lvl] = b
                for d in range(lvl + 1, 5):
                    active[d] = None
                df2.at[i, out_selected_col] = pd.NA
            else:
                selected = any(v is True for v in active)
                df2.at[i, out_selected_col] = selected

    return df2


def _nearest_context_title(df_law: pd.DataFrame, row_pos: int) -> str:
    """
    Optional: find the most recent non-article label above this article, for context.
    """
    # row_pos is positional index within df_law (already sorted)
    for j in range(row_pos - 1, -1, -1):
        if int(df_law.iloc[j]["level"]) != 5:
            label = str(df_law.iloc[j].get("label", "") or "").strip()
            if label:
                return label
    return ""


def classify_selected_articles(
    client: LocalLLMClient,
    df: pd.DataFrame,
    cfg: Run2Config,
    title_relevant_col: str = "title_ai_relevant",
) -> pd.DataFrame:
    """
    Adds:
    - cfg.out_selected_col: bool for articles, NA for non-articles
    - cfg.out_ai_relevant_col: bool for articles selected, NA otherwise
    - cfg.out_targets_col: JSON string list for articles selected, NA otherwise
    - cfg.out_justification_col: string for articles selected, NA otherwise
    """
    df2 = compute_run2_selection(
        df=df,
        title_relevant_col=title_relevant_col,
        out_selected_col=cfg.out_selected_col,
    ).copy()

    # init outputs
    for col in [cfg.out_ai_relevant_col, cfg.out_targets_col, cfg.out_justification_col]:
        if col not in df2.columns:
            df2[col] = pd.NA

    df2 = _stable_sort(df2)

    # classify per law, per article
    for law_id, df_law in df2.groupby("law_id", sort=False):
        df_law = df_law.reset_index()  # keeps original index in df2 as "index"

        for pos in range(len(df_law)):
            row = df_law.iloc[pos]
            lvl = int(row["level"])
            if lvl != 5:
                continue

            selected = row[cfg.out_selected_col]
            if selected is not True:
                continue

            original_idx = int(row["index"])

            if cfg.skip_if_already_done and pd.notna(df2.at[original_idx, cfg.out_ai_relevant_col]):
                continue

            article_text = str(row.get("text", "") or "").strip()
            if not article_text:
                # empty article text -> mark as not relevant (conservative)
                df2.at[original_idx, cfg.out_ai_relevant_col] = False
                df2.at[original_idx, cfg.out_targets_col] = json.dumps([], ensure_ascii=False)
                df2.at[original_idx, cfg.out_justification_col] = "Texte de l'article vide."
                continue

            context_title = _nearest_context_title(df_law, pos)

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_PROMPT_TEMPLATE.format(
                        law_id=str(law_id),
                        node_id=str(row.get("node_id", "")),
                        row_uid=int(row.get("row_uid")),
                        context_title=context_title if context_title else "(aucun)",
                        article_text=article_text,
                    ),
                },
            ]

            raw = client.chat(
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                response_format={"type": "json_object"},
            )

            data = _safe_json_loads(raw)

            ai_rel = bool(data.get("ai_relevant", False))
            targets = data.get("targets", [])
            if not isinstance(targets, list):
                targets = []
            # normalize allowed targets
            allowed = {"Data", "Computing Infrastructure", "Development & Adoption", "Skills"}
            targets_norm = [t for t in targets if isinstance(t, str) and t in allowed]
            if not ai_rel:
                targets_norm = []

            justif = str(data.get("justification", "") or "").strip()
            if not justif:
                justif = "Aucune justification fournie."

            df2.at[original_idx, cfg.out_ai_relevant_col] = ai_rel
            df2.at[original_idx, cfg.out_targets_col] = json.dumps(targets_norm, ensure_ascii=False)
            df2.at[original_idx, cfg.out_justification_col] = justif

    return df2
