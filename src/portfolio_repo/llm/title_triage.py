from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from portfolio_repo.llm.client import LLMClient


@dataclass(frozen=True)
class TriageConfig:
    # LLM params
    temperature: float = 0.0
    max_tokens: int = 800

    # batching
    chunk_size: int = 80

    # output column name
    out_col: str = "title_ai_relevant"

    # if True, keep existing values in out_col (resume behavior)
    skip_if_already_done: bool = True

    # debug
    debug_prompt: bool = False
    debug_raw_response: bool = False
    debug_max_chars: int = 4000


_KEYWORDS_BLOCK = (
    "LISTE DE MOTS-CLÉS (match simple dans le libellé ; casse/accents peuvent varier) :\n"
    "- automatis\n"
    "- traitement automatis\n"
    "- système automatis\n"
    "- algorith\n"
    "- intelligence artific\n"
)


def _system_prompt() -> str:
    return (
        "Tu analyses une liste de TITRES juridiques appartenant à UNE seule loi.\n\n"
        "Tâche : sélectionner les titres en lien avec des systèmes automatisés/informatiques, "
        "le traitement de données, des algorithmes, ou l’IA.\n\n"
        "RÈGLE STRICTE (obligatoire) :\n"
        "- Tu n’as le droit de sélectionner un titre QUE SI le libellé contient explicitement au moins un mot/fragment "
        "de la LISTE DE MOTS-CLÉS fournie.\n"
        "- Pour chaque titre sélectionné, la justification doit contenir le ou les fragments EXACTS copiés depuis le libellé "
        "(entre guillemets). Pas de paraphrase.\n"
        "- Si tu ne peux pas citer un fragment exact présent dans le libellé, NE SÉLECTIONNE PAS.\n\n"
        "Retourne UNIQUEMENT du JSON strict.\n"
    )


def _user_prompt(law_id: str, titles_block: str) -> str:
    return (
        f"Loi ID: {law_id}\n\n"
        f"{_KEYWORDS_BLOCK}\n"
        "TITRES (format: row_uid. libellé)\n"
        f"{titles_block}\n\n"
        "FORMAT EXACT DE RÉPONSE (JSON strict) :\n"
        "{\n"
        '  "selected": [\n'
        "    {\n"
        '      "row_uid": 123,\n'
        '      "selected": true,\n'
        '      "justification": "..." \n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _safe_parse_selected_json(raw: str) -> List[Dict[str, Any]]:
    """
    Extract expected JSON:
      {"selected": [{"row_uid":..., "selected":..., "justification":...}, ...]}
    Tries strict json first, then finds first {...} block.
    """
    raw = (raw or "").strip()

    # strict
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("selected"), list):
            return obj["selected"]
    except Exception:
        pass

    # try to find JSON block
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("selected"), list):
                return obj["selected"]
        except Exception:
            pass

    return []


def run_title_triage_for_law(
    client: LLMClient,
    df_law_non_articles: pd.DataFrame,
    cfg: TriageConfig,
) -> Dict[int, bool]:
    """
    Input: dataframe d'UNE seule loi, avec level != 5 (titres/sections uniquement)
    Output: mapping row_uid -> bool
    """
    if df_law_non_articles.empty:
        return {}

    law_ids = df_law_non_articles["law_id"].unique().tolist()
    if len(law_ids) != 1:
        raise ValueError(f"Expected exactly 1 law_id, got {law_ids}")
    law_id = str(law_ids[0])

    # (row_uid, label)
    rows: List[Tuple[int, str]] = list(
        zip(
            df_law_non_articles["row_uid"].astype(int).tolist(),
            df_law_non_articles["label"].astype(str).tolist(),
        )
    )

    selected: Dict[int, bool] = {}

    for i in range(0, len(rows), cfg.chunk_size):
        chunk = rows[i : i + cfg.chunk_size]
        titles_block = "\n".join([f"{uid}. {label}" for uid, label in chunk])

        sys_p = _system_prompt()
        usr_p = _user_prompt(law_id=law_id, titles_block=titles_block)

        if cfg.debug_prompt:
            print("\n" + "=" * 90)
            print("[RUN1 DEBUG] law_id=", law_id, "chunk", i, "-", i + len(chunk))
            print("-" * 90)
            print("[SYSTEM]\n" + sys_p)
            print("-" * 90)
            print("[USER]\n" + usr_p[: cfg.debug_max_chars])
            if len(usr_p) > cfg.debug_max_chars:
                print("... (truncated)")
            print("=" * 90 + "\n")

        raw = client.chat(
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        if cfg.debug_raw_response:
            print("\n" + "-" * 90)
            print("[RUN1 DEBUG RAW RESPONSE]\n" + (raw or "")[: cfg.debug_max_chars])
            if raw and len(raw) > cfg.debug_max_chars:
                print("... (truncated)")
            print("-" * 90 + "\n")

        parsed = _safe_parse_selected_json(raw)

        # default False for all chunk items
        chunk_uids = {uid for uid, _ in chunk}
        for uid in chunk_uids:
            selected[uid] = False

        for item in parsed:
            try:
                uid = int(item.get("row_uid"))
            except Exception:
                continue
            if uid in chunk_uids:
                selected[uid] = bool(item.get("selected", False))

    return selected


def triage_titles_dataset(
    client: LLMClient,
    df: pd.DataFrame,
    cfg: TriageConfig,
) -> pd.DataFrame:
    """
    Applies title triage to a multi-law dataset.

    - Creates row_uid if absent.
    - Leaves level==5 (articles) untouched (NA in out_col).
    - Writes boolean selections for level != 5 into cfg.out_col.
    """
    df_out = df.copy()

    if "row_uid" not in df_out.columns:
        df_out["row_uid"] = range(1, len(df_out) + 1)

    # Resume behavior: if column exists and skip is enabled, keep existing values
    if cfg.out_col not in df_out.columns:
        df_out[cfg.out_col] = pd.NA
    elif cfg.skip_if_already_done:
        # keep existing; only fill missing
        pass
    else:
        df_out[cfg.out_col] = pd.NA

    # work only on non-articles
    df_non_articles = df_out[df_out["level"] != 5].copy()

    # if resuming, we only process rows where out_col is NA
    if cfg.skip_if_already_done:
        df_non_articles = df_non_articles[df_non_articles[cfg.out_col].isna()].copy()

    # nothing to do
    if df_non_articles.empty:
        return df_out

    # group by law_id
    for law_id, df_law in df_non_articles.groupby("law_id", sort=False):
        mapping = run_title_triage_for_law(client=client, df_law_non_articles=df_law, cfg=cfg)

        # assign back only for these rows (level != 5 and same law_id and NA if resume)
        mask = df_out["law_id"].eq(law_id) & (df_out["level"] != 5)
        if cfg.skip_if_already_done:
            mask = mask & df_out[cfg.out_col].isna()

        df_out.loc[mask, cfg.out_col] = (
            df_out.loc[mask, "row_uid"].astype(int).map(mapping).fillna(False).astype(bool)
        )

    return df_out
