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
    out_targets_col: str = "article_targets"  # JSON string list
    out_justification_col: str = "article_justification"

    # if True, skip already-classified articles (resume behavior)
    skip_if_already_done: bool = True

    # NEW: debug
    debug_prompt: bool = False
    debug_raw_response: bool = False
    debug_max_chars: int = 6000


_SYSTEM_PROMPT = (
    "Rôle: codeur juridique.\n\n"
    "Définition de l'IA dans ce contexte:\n"
    "- Tout système, logiciel, procédé ou dispositif exécutant des fonctions de manière automatisée "
    "en se basant sur de grosses quantités de données ou exploitant de la prise de décision algorithmique peuvent être considérée ici comme de l'IA.\n\n"
    "Règle d'imputation juridique (OBLIGATOIRE):\n"
    "- Si un article régule un objet X équipé ou utilisant un système automatisé Y, "
    "alors l'article régule aussi le système Y dès lors qu'il fixe des conditions "
    "d'usage, d'admission, de surveillance, d'autorisation, d'essais ou d'obligations.\n\n"
    "Targets possibles (définitions STRICTES):\n"
    "- Development & Adoption: règles juridiques portant sur l'usage, l'admission, "
    "le déploiement, les essais, la supervision ou la responsabilité d'un système automatisé.\n"
    "- Data: règles juridiques portant sur des données destinées à être traitées, produites "
    "ou générées par un système automatisé (collecte, accès, journalisation, transmission, effacement).\n"
    "- Computing Infrastructure: exigences techniques de sécurité, d'accès, de stockage, "
    "d'intégrité ou de communication applicables à un système informatique.\n"
    "- Skills: Toutes règles juridiques ayant un impact sur le développement de compétences numériques / IA.\n\n"
    "Règles de décision (STRICTES):\n"
    "- Si AUCUN système automatisé n'est détecté → ai_relevant = false.\n"
    "- Si un système automatisé est détecté ET que l'article fixe des règles sur son usage, "
    "son admission ou sa supervision → ai_relevant = true ET targets NE PEUT PAS être vide.\n"
    "- N'inclus une target QUE si elle est clairement justifiée par le texte.\n"
    "- Il est tout a fait possible d'avoir plusieurs targets si nécessaire.\n\n"
    "Sortie: UNIQUEMENT du JSON valide. Aucune explication hors JSON."
)


_USER_PROMPT_TEMPLATE = """Analyse UN article. Retour JSON strict uniquement.

IDs:
- law_id: {law_id}
- node_id: {node_id}
- row_uid: {row_uid}

Contexte (titre le plus proche au-dessus, si disponible):
- context_title: {context_title}

Texte:
\"\"\"{article_text}\"\"\"

Tâche (ordre OBLIGATOIRE):
1) DÉTECTION:
   - Un système automatisé/algorithmique est-il présent ? (true/false)
   - Si true, cite le terme déclencheur EXACT du texte.

2) CLASSIFICATION:
   - ai_relevant (true/false)
   - targets (liste parmi: Development & Adoption, Data, Computing Infrastructure, Skills)
   - justification concise (1–2 phrases, fondée uniquement sur le texte)

Contraintes STRICTES:
- Si ai_relevant = false → targets = []
- Si ai_relevant = true → targets NE PEUT PAS être vide
- N'inclus une target QUE si le texte la justifie explicitement
- Tu peux inclure plusieurs targets si nécessaire

Format EXACT (structure uniquement — NE PAS recopier les valeurs d’exemple):
{{
  "row_uid": {row_uid},
  "automation_present": <true|false>,
  "automation_trigger": "<extrait exact ou vide>",
  "ai_relevant": <true|false>,
  "targets": ["<0..4 parmi: Development & Adoption, Data, Computing Infrastructure, Skills>"],
  "justification": "<1–2 phrases>"
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

    for col in [cfg.out_ai_relevant_col, cfg.out_targets_col, cfg.out_justification_col]:
        if col not in df2.columns:
            df2[col] = pd.NA

    df2 = _stable_sort(df2)

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

            if cfg.debug_prompt:
                print("\n" + "=" * 90)
                print(
                    f"[RUN2 DEBUG] law_id={law_id} row_uid={int(row.get('row_uid'))} node_id={row.get('node_id')}"
                )
                print("-" * 90)
                for m in messages:
                    role = m.get("role", "")
                    content = str(m.get("content", "") or "")
                    if len(content) > cfg.debug_max_chars:
                        content = content[: cfg.debug_max_chars] + "\n... [TRUNCATED]"
                    print(f"\n[{role.upper()}]\n{content}")
                print("=" * 90 + "\n")

            raw = client.chat(
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                response_format={"type": "json_object"},
            )

            if cfg.debug_raw_response:
                preview = raw if len(raw) <= cfg.debug_max_chars else raw[: cfg.debug_max_chars] + "\n... [TRUNCATED]"
                print("\n" + "-" * 90)
                print("[RUN2 DEBUG] RAW MODEL OUTPUT")
                print(preview)
                print("-" * 90 + "\n")

            data = _safe_json_loads(raw)

            ai_rel = bool(data.get("ai_relevant", False))
            targets = data.get("targets", [])
            if not isinstance(targets, list):
                targets = []

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
