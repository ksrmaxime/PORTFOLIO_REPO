# src/portfolio_repo/llm/title_triage.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from portfolio_repo.llm.client import LocalLLMClient


@dataclass(frozen=True)
class TriageConfig:
    chunk_size: int = 80
    temperature: float = 0.0
    max_tokens: int = 800
    # colonne de sortie (remplie uniquement pour level != 5)
    out_col: str = "title_ai_relevant"


_SYSTEM_PROMPT = (
    "Tu analyses des TITRES/INTITULÉS (pas le texte des articles). "
    "Ta tâche: dire si le titre suggère un lien direct avec: systèmes automatisés/algorithmes, "
    "traitement automatisé de données, infrastructures informatiques/de calcul (serveurs, cloud), "
    "ou intelligence artificielle.\n\n"
    "Sois CONSERVATEUR: si ce n’est pas clair dans le titre, réponds FALSE.\n"
    "Tu ne dois PAS déduire à partir du secteur (ex: santé, aviation) si le titre ne mentionne rien d’automatisé/data/IT.\n\n"
    "Retourne UNIQUEMENT du JSON valide, sans texte autour."
)

_USER_PROMPT_TEMPLATE = """Tu reçois une liste de lignes, toutes issues d'une seule loi (law_id={law_id}).
Pour chaque ligne, réponds TRUE ou FALSE à la question:
"Ce titre est-il en lien avec systèmes automatisés / algorithmes / traitement automatisé de données / infrastructures informatiques ou de calcul / intelligence artificielle ?"

Règles:
- Base-toi UNIQUEMENT sur le champ "label".
- Si doute -> FALSE.
- JSON strict uniquement.

Entrée:
{items_json}

Format de sortie EXACT:
{{
  "items": [
    {{"row_uid": 123, "ai_relevant": true}},
    {{"row_uid": 124, "ai_relevant": false}}
  ]
}}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")  # ", }" or ", ]"
_BOOL_FIXES = [
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
]

# fallback extraction: {"row_uid": 123, "ai_relevant": true}
_PAIR_RE = re.compile(
    r'"row_uid"\s*:\s*(\d+)\s*,\s*"ai_relevant"\s*:\s*(true|false)',
    flags=re.IGNORECASE,
)


def _try_repair_json(text: str) -> str:
    t = text.strip()

    # keep only first {...} block if any surrounding text exists
    m = _JSON_BLOCK_RE.search(t)
    if m:
        t = m.group(0).strip()

    # common repairs
    for rx, repl in _BOOL_FIXES:
        t = rx.sub(repl, t)

    # remove trailing commas
    t = _TRAILING_COMMA_RE.sub(r"\1", t)

    # sometimes model uses single quotes (rare but happens)
    # do this conservatively only if there are no double quotes at all
    if '"' not in t and "'" in t:
        t = t.replace("'", '"')

    return t


def _safe_parse_items_mapping(raw: str) -> Dict[int, bool]:
    """
    Ultimate fallback: parse row_uid + ai_relevant pairs by regex,
    even if JSON is invalid.
    """
    out: Dict[int, bool] = {}
    for uid_s, val_s in _PAIR_RE.findall(raw):
        uid = int(uid_s)
        val = val_s.lower() == "true"
        out[uid] = val
    return out


def _safe_json_loads(text: str) -> dict:
    t = _try_repair_json(text)
    return json.loads(t)



def _iter_chunks(rows: List[Tuple[int, str]], chunk_size: int) -> Iterable[List[Tuple[int, str]]]:
    for i in range(0, len(rows), chunk_size):
        yield rows[i : i + chunk_size]


def build_messages(law_id: str, chunk_rows: List[Tuple[int, str]]) -> List[Dict[str, str]]:
    items = [{"row_uid": int(uid), "label": (label or "").strip()} for uid, label in chunk_rows]
    user_prompt = _USER_PROMPT_TEMPLATE.format(law_id=law_id, items_json=json.dumps(items, ensure_ascii=False))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def run_title_triage_for_law(
    client: LocalLLMClient,
    df_law_non_articles: pd.DataFrame,
    cfg: TriageConfig,
) -> Dict[int, bool]:
    """
    Input: df filtré à une seule loi, et level != 5.
    Output: mapping row_uid -> bool
    """
    law_ids = df_law_non_articles["law_id"].unique().tolist()
    if len(law_ids) != 1:
        raise ValueError(f"Expected exactly 1 law_id, got {law_ids}")
    law_id = str(law_ids[0])

    rows: List[Tuple[int, str]] = list(
        zip(
            df_law_non_articles["row_uid"].astype(int).tolist(),
            df_law_non_articles["label"].astype(str).tolist(),
        )
    )

    out: Dict[int, bool] = {}

    for chunk in _iter_chunks(rows, cfg.chunk_size):
        messages = build_messages(law_id=law_id, chunk_rows=chunk)
        raw = client.chat(
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            data = _safe_json_loads(raw)
            items = data.get("items")
            if not isinstance(items, list):
                raise ValueError("missing items list")

            for it in items:
                uid = int(it["row_uid"])
                val = bool(it["ai_relevant"])
                out[uid] = val

        except Exception as e:
            # fallback regex extraction
            recovered = _safe_parse_items_mapping(raw)
            if recovered:
                out.update(recovered)
            else:
                # if nothing recoverable, raise with useful context
                preview = raw[:2000].replace("\n", "\\n")
                raise RuntimeError(
                    f"Failed to parse LLM JSON for law_id={law_id}. "
                    f"Error={type(e).__name__}: {e}. "
                    f"Raw preview (first 2000 chars): {preview}"
                ) from e

    return out


def add_row_uid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un ID stable si on trie d'abord.
    """
    sort_cols = [c for c in ["law_id", "order_index", "level", "node_id"] if c in df.columns]
    df2 = df.sort_values(sort_cols, kind="stable").reset_index(drop=True).copy()
    df2["row_uid"] = range(1, len(df2) + 1)
    return df2


def triage_titles_dataset(
    client: LocalLLMClient,
    df: pd.DataFrame,
    cfg: TriageConfig,
) -> pd.DataFrame:
    """
    - Ajoute row_uid
    - Envoie au LLM uniquement level != 5 (non-articles), et uniquement label (dans le prompt)
    - Ajoute une colonne cfg.out_col (NaN pour level == 5)
    """
    df2 = add_row_uid(df)

    # init colonne sortie vide
    df2[cfg.out_col] = pd.NA

    df_non_articles = df2[df2["level"] != 5].copy()
    # group-by law_id pour respecter "une seule loi" dans le prompt
    for law_id, grp in df_non_articles.groupby("law_id", sort=False):
        mapping = run_title_triage_for_law(client=client, df_law_non_articles=grp, cfg=cfg)
        mask = (df2["law_id"] == law_id) & (df2["level"] != 5)
        df2.loc[mask, cfg.out_col] = df2.loc[mask, "row_uid"].map(mapping)

    return df2
