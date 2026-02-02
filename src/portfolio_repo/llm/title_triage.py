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

    # NEW: to avoid silent "all False" when parsing fails
    fail_on_unparseable_response: bool = True


def _system_prompt() -> str:
    return (
        "Tu analyses des TITRES/INTITULÉS (pas le texte des articles). "
        "Ta tâche: dire si le titre suggère un lien direct avec: systèmes automatisés/algorithmes, "
        "traitement automatisé de données, infrastructures informatiques/de calcul (serveurs, cloud), "
        "ou intelligence artificielle.\n\n"
        "Sois CONSERVATEUR: si ce n’est pas clair dans le titre, réponds FALSE.\n"
        "Tu ne dois PAS déduire à partir du secteur (ex: santé, aviation) "
        "si le titre ne mentionne rien d’automatisé/data/IT.\n\n"
        "Retourne UNIQUEMENT du JSON valide, sans texte autour."
    )


def _user_prompt(law_id: str, items_json: str) -> str:
    return f"""Tu reçois une liste de lignes, toutes issues d'une seule loi (law_id={law_id}).
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


def _extract_first_json_candidate(raw: str) -> Optional[str]:
    """
    Returns a JSON-looking substring, or None.
    - First try raw stripped.
    - Then try first {...} block.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        return m.group(0)
    return None


def _repair_json_text(s: str) -> str:
    """
    Best-effort repairs for common LLM JSON issues:
    - Python booleans/None -> JSON booleans/null
    - trailing commas
    - smart quotes
    - conservative single-quote conversion for keys/simple string values
    """
    s = (s or "").strip()
    if not s:
        return s

    # smart quotes -> normal quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Python literals -> JSON literals
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)

    # trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Conservative single-quote conversion:
    # keys: {'items': ...} -> {"items": ...}
    s = re.sub(r"(?P<prefix>[{,\s])'(?P<key>[A-Za-z0-9_]+)'\s*:", r'\g<prefix>"\g<key>":', s)
    # simple string values: "justification": 'abc' -> "justification": "abc"
    s = re.sub(r":\s*'(?P<val>[^'\n\r]*)'\s*(?P<suffix>[,}\]])", r': "\g<val>"\g<suffix>', s)

    return s


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    """
    Parse a JSON dict with multiple fallbacks + repairs.
    Returns {} if it can't parse a dict.
    """
    cand = _extract_first_json_candidate(raw)
    if not cand:
        return {}

    # Attempt 1: direct
    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Attempt 2: repaired
    repaired = _repair_json_text(cand)
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _fallback_pairs_from_regex(raw: str) -> Dict[int, bool]:
    """
    Ultra-robust fallback: extract (row_uid, ai_relevant) pairs even if JSON is broken.
    Accepts patterns like:
      row_uid: 123 ... ai_relevant: true/false
    in any order, across the text.
    """
    text = (raw or "")
    if not text:
        return {}

    # normalize python booleans to json booleans for matching
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)

    pairs: Dict[int, bool] = {}

    # First try to extract per-item blocks like { ... } (most precise)
    blocks = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not blocks:
        blocks = text.splitlines()

    row_pat = re.compile(r"row_uid\s*[:=]\s*(\d+)")
    rel_pat = re.compile(r"ai_relevant\s*[:=]\s*(true|false)")

    for b in blocks:
        m_uid = row_pat.search(b)
        m_rel = rel_pat.search(b)
        if m_uid and m_rel:
            try:
                uid = int(m_uid.group(1))
                val = (m_rel.group(1) == "true")
                pairs[uid] = val
            except Exception:
                continue

    # If still empty, do a broader scan across the full text (less strict)
    if not pairs:
        broad = re.findall(
            r"row_uid\s*[:=]\s*(\d+).*?ai_relevant\s*[:=]\s*(true|false)",
            text,
            flags=re.DOTALL,
        )
        for uid_s, val_s in broad:
            try:
                pairs[int(uid_s)] = (val_s == "true")
            except Exception:
                continue

    return pairs


def _safe_parse_items_mapping(raw: str) -> Dict[int, bool]:
    """
    Preferred parse path:
      {"items": [{"row_uid": 123, "ai_relevant": true}, ...]}
    Fallback:
      regex extraction of row_uid + ai_relevant from semi-structured output.
    """
    obj = _safe_json_loads(raw)
    mapping: Dict[int, bool] = {}

    items = obj.get("items")
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                uid = int(it.get("row_uid"))
            except Exception:
                continue
            mapping[uid] = bool(it.get("ai_relevant", False))

    if mapping:
        return mapping

    return _fallback_pairs_from_regex(raw)


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

    rows: List[Tuple[int, str]] = list(
        zip(
            df_law_non_articles["row_uid"].astype(int).tolist(),
            df_law_non_articles["label"].astype(str).tolist(),
        )
    )

    selected: Dict[int, bool] = {}

    for i in range(0, len(rows), cfg.chunk_size):
        chunk = rows[i : i + cfg.chunk_size]
        items = [{"row_uid": int(uid), "label": (label or "").strip()} for uid, label in chunk]

        sys_p = _system_prompt()
        usr_p = _user_prompt(
            law_id=law_id,
            items_json=json.dumps(items, ensure_ascii=False),
        )

        if cfg.debug_prompt:
            print("\n" + "=" * 90)
            print(f"[RUN1 DEBUG] law_id={law_id} chunk {i}-{i + len(chunk)}")
            print("-" * 90)
            print("[SYSTEM]\n" + sys_p)
            print("-" * 90)
            print("[USER]\n" + usr_p[: cfg.debug_max_chars])
            if len(usr_p) > cfg.debug_max_chars:
                print("... (truncated)")
            print("=" * 90 + "\n")

        # Try JSON mode if: if your backend doesn't support it, fallback without it.
        try:
            raw = client.chat(
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": usr_p},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                response_format={"type": "json_object"},
            )
        except TypeError:
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

        parsed_map = _safe_parse_items_mapping(raw)

        chunk_uids = {uid for uid, _ in chunk}

        # Default False for chunk, then fill True where parsed
        for uid in chunk_uids:
            selected[uid] = False

        if cfg.fail_on_unparseable_response and not parsed_map:
            preview = (raw or "")[: cfg.debug_max_chars]
            raise ValueError(
                "LLM response unparseable (no row_uid/ai_relevant pairs recovered). "
                "Set debug_raw_response=True to inspect.\n"
                f"RAW PREVIEW:\n{preview}"
            )

        for uid, val in parsed_map.items():
            if uid in chunk_uids:
                selected[uid] = bool(val)

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

    if cfg.out_col not in df_out.columns:
        df_out[cfg.out_col] = pd.NA
    elif not cfg.skip_if_already_done:
        df_out[cfg.out_col] = pd.NA

    df_non_articles = df_out[df_out["level"] != 5].copy()

    if cfg.skip_if_already_done:
        df_non_articles = df_non_articles[df_non_articles[cfg.out_col].isna()].copy()

    if df_non_articles.empty:
        return df_out

    for law_id, df_law in df_non_articles.groupby("law_id", sort=False):
        mapping = run_title_triage_for_law(client=client, df_law_non_articles=df_law, cfg=cfg)

        mask = df_out["law_id"].eq(law_id) & (df_out["level"] != 5)
        if cfg.skip_if_already_done:
            mask = mask & df_out[cfg.out_col].isna()

        df_out.loc[mask, cfg.out_col] = (
            df_out.loc[mask, "row_uid"].astype(int).map(mapping).fillna(False).astype(bool)
        )

    return df_out



