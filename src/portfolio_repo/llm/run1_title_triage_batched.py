# src/portfolio_repo/llm/run1_title_triage_batched.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from portfolio_repo.llm.curnagl_client import TransformersClient


@dataclass(frozen=True)
class Run1Config:
    out_col: str = "title_ai_relevant"
    out_just_col: str = "title_ai_justification"

    # how many titles per prompt
    items_per_prompt: int = 40

    # how many prompts per GPU generate() call
    prompts_per_batch: int = 8

    # LLM params
    temperature: float = 0.0
    max_tokens: int = 220

    # resume behavior
    skip_if_already_done: bool = True


# --- Prompt V4 (base) + ajout: justification obligatoire pour chaque TRUE ---
_SYSTEM_PROMPT = (
    "Tu reçois ci-dessous une liste de titres, sous-titres, chapitres etc. "
    "représentant la structure des textes de lois suisses.\n"
    "Tu n'as accès qu'aux TITRES/INTITULÉS, pas au texte des articles.\n\n"

    "Tâche: identifier les titres qui pourraient être à la tête d'une section "
    "contenant des articles en lien avec l'intelligence artificielle.\n"
    "L'IA est définie de manière large: systèmes automatisés/algorithmes, "
    "traitement automatisé de données, infrastructures informatiques/de calcul "
    "(serveurs, cloud), ainsi que l'IA au sens strict.\n\n"

    "Important:\n"
    "- Base-toi UNIQUEMENT sur les mots du titre.\n"
    "- N'infère pas le contenu des articles.\n"
    "- Les titres génériques ou structurels (ex: Preuve, Définitions, Chapitre X, "
    "Dispositions finales) ne doivent PAS être sélectionnés sauf s'ils "
    "contiennent un indice explicite.\n"
    "- Pour chaque titre sélectionné, tu dois fournir une justification courte.\n\n"

    "Réponds UNIQUEMENT avec ce JSON strict:\n"
    "{\n"
    "  \"true_row_uids\": [],\n"
    "  \"justifications\": {}\n"
    "}\n"
    "en remplaçant [] par les row_uid jugés pertinents.\n"
    "Pour chaque row_uid sélectionné, ajoute une justification courte dans \"justifications\" "
    "(clé = row_uid, valeur = justification).\n"
    "Aucun autre texte."

)


def _make_user_prompt(rows: List[Tuple[int, str]]) -> str:
    lines = "\n".join(f"{uid}\t{label.strip()}" for uid, label in rows)
    return "Données (row_uid<TAB>label):\n" + lines


def _extract_last_json_object(text: str) -> Optional[str]:
    """
    Extract the last top-level JSON object {...} from a string, even if surrounded by text.
    """
    if not text:
        return None

    s = text.strip().replace("“", '"').replace("”", '"')
    # common trailing comma cleanup inside JSON (best-effort)
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # scan for balanced braces and keep the last complete object
    last_obj = None
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last_obj = s[start : i + 1]
                    start = None
    return last_obj


def _parse_uid_and_justifications(raw: str) -> tuple[list[int], dict[int, str]]:
    raw = (raw or "").strip()
    if not raw:
        return [], {}

    jtxt = _extract_last_json_object(raw)
    if not jtxt:
        return [], {}

    try:
        obj = json.loads(jtxt)
    except Exception:
        return [], {}

    uids: list[int] = []
    justs: dict[int, str] = {}

    # uids
    for uid in obj.get("true_row_uids", []):
        try:
            uids.append(int(uid))
        except Exception:
            continue

    # justifications
    jmap = obj.get("justifications", {})
    if isinstance(jmap, dict):
        for k, v in jmap.items():
            try:
                uid = int(k)
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                justs[uid] = v.strip()

    return uids, justs



def run1_title_triage_batched(
    client: TransformersClient,
    df: pd.DataFrame,
    cfg: Run1Config,
) -> pd.DataFrame:
    df_out = df.copy()

    if "row_uid" not in df_out.columns:
        df_out["row_uid"] = range(1, len(df_out) + 1)

    # output columns init/reset
    if cfg.out_col not in df_out.columns:
        df_out[cfg.out_col] = pd.NA
    elif not cfg.skip_if_already_done:
        df_out[cfg.out_col] = pd.NA

    if cfg.out_just_col not in df_out.columns:
        df_out[cfg.out_just_col] = pd.NA
    elif not cfg.skip_if_already_done:
        df_out[cfg.out_just_col] = pd.NA

    # Triages only levels 1..4 (excludes law title=0 and articles=5)
    df_non = df_out[df_out["level"].isin([1, 2, 3, 4])].copy()
    if cfg.skip_if_already_done:
        df_non = df_non[df_non[cfg.out_col].isna()].copy()

    if df_non.empty:
        return df_out

    rows: List[Tuple[int, str]] = list(
        zip(df_non["row_uid"].astype(int).tolist(), df_non["label"].astype(str).tolist())
    )

    selected_map: Dict[int, bool] = {}
    just_map: Dict[int, Optional[str]] = {}

    # Build prompts
    prompts: List[str] = []
    prompt_chunks: List[List[Tuple[int, str]]] = []
    for i in range(0, len(rows), cfg.items_per_prompt):
        chunk = rows[i : i + cfg.items_per_prompt]
        prompts.append(_make_user_prompt(chunk))
        prompt_chunks.append(chunk)

    # Process prompts in GPU batches
    for j in range(0, len(prompts), cfg.prompts_per_batch):
        batch_prompts = prompts[j : j + cfg.prompts_per_batch]
        batch_chunks = prompt_chunks[j : j + cfg.prompts_per_batch]

        if j == 0:
            print("\n=== SYSTEM PROMPT ===")
            print(_SYSTEM_PROMPT)
            print("\n=== USER PROMPT (first batch, first prompt) ===")
            print(batch_prompts[0][:3000])

        raws = client.chat_many(
            system_prompt=_SYSTEM_PROMPT,
            user_prompts=batch_prompts,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        if j == 0:
            print("\n=== RAW LLM RESPONSE (first batch, first prompt) ===")
            print((raws[0] or "")[:3000])

        for raw, chunk in zip(raws, batch_chunks):
            uids = [uid for uid, _ in chunk]
            allowed = set(uids)

            true_uids, justs = _parse_uid_and_justifications(raw)

            # retry once if totally unparseable (often prompt-echo)
            if not true_uids and not justs:
                retry_system = _SYSTEM_PROMPT + "\n\nRAPPEL CRITIQUE: réponds uniquement avec le JSON, sans répéter le prompt."
                retry_raw = client.chat_many(
                    system_prompt=retry_system,
                    user_prompts=[_make_user_prompt(chunk)],
                    temperature=0.0,
                    max_tokens=cfg.max_tokens,
                )[0]
                true_uids, justs = _parse_uid_and_justifications(retry_raw)

            # default values
            for uid in uids:
                selected_map[uid] = False
                just_map[uid] = pd.NA

            # assign
            for uid in true_uids:
                if uid not in allowed:
                    continue
                selected_map[uid] = True
                j = justs.get(uid, "")
                just_map[uid] = j if j else pd.NA

            # log if still unparseable after retry
            if (not true_uids and not justs) and cfg.debug_raw_response:
                print("[RUN1 WARN] Unparseable response; chunk forced FALSE. Raw preview:")
                print((raw or "")[:800])


    # Write results back (only levels 1..4)
    mask = df_out["level"].isin([1, 2, 3, 4])
    if cfg.skip_if_already_done:
        mask = mask & df_out[cfg.out_col].isna()

    df_out.loc[mask, cfg.out_col] = (
        df_out.loc[mask, "row_uid"].astype(int).map(selected_map).fillna(False).astype(bool)
    )

    # justification only for TRUE (else NA)
    def _just_or_na(uid: int, is_true: bool) -> object:
        if not is_true:
            return pd.NA
        return just_map.get(uid, pd.NA)

    tmp_uids = df_out.loc[mask, "row_uid"].astype(int)
    tmp_true = df_out.loc[mask, cfg.out_col].astype(bool)
    df_out.loc[mask, cfg.out_just_col] = [
        _just_or_na(int(uid), bool(t)) for uid, t in zip(tmp_uids.tolist(), tmp_true.tolist())
    ]

    return df_out
