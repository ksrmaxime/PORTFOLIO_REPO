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
    "Remplace [] par les row_uid jugés pertinents.\n"
    "Pour chaque row_uid sélectionné, ajoute une justification courte dans "
    "\"justifications\" (clé = row_uid, valeur = justification).\n"
    "Ne mets PAS de justification pour les row_uid non sélectionnés.\n"
    "Aucun autre texte."
)


def _make_user_prompt(rows: List[Tuple[int, str]]) -> str:
    lines = "\n".join(f"{uid}\t{label.strip()}" for uid, label in rows)
    return "Données (row_uid<TAB>label):\n" + lines


def _parse_true_rows(raw: str) -> List[Tuple[int, str]]:
    """
    Parse expected JSON:
      {"true_rows":[{"row_uid":123,"justification":"..."}]}
    Returns list of (row_uid, justification).
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    # take last json object containing "true_rows"
    candidates = re.findall(r"\{[\s\S]*?\"true_rows\"[\s\S]*?\}", raw)
    cand = candidates[-1] if candidates else raw

    cand = cand.replace("“", '"').replace("”", '"')
    cand = re.sub(r",\s*([}\]])", r"\1", cand)

    try:
        obj = json.loads(cand)
    except Exception:
        # fallback: try to locate true_rows array as text (weak fallback)
        return []

    rows = obj.get("true_rows", [])
    out: List[Tuple[int, str]] = []
    if not isinstance(rows, list):
        return out

    for r in rows:
        if not isinstance(r, dict):
            continue
        uid = r.get("row_uid", None)
        just = r.get("justification", "")
        try:
            uid_i = int(uid)
        except Exception:
            continue
        if not isinstance(just, str):
            just = ""
        just = just.strip()
        out.append((uid_i, just))
    return out


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

            parsed = _parse_true_rows(raw)

            # defaults
            for uid in uids:
                selected_map[uid] = False
                just_map[uid] = pd.NA

            # assign truths with justifications
            got_any_key = "true_rows" in (raw or "")
            for uid, just in parsed:
                if uid not in allowed:
                    continue
                selected_map[uid] = True
                just_map[uid] = just if just else pd.NA

            # guardrail: if totally unparseable (and doesn't even mention the key), fail fast
            if not got_any_key:
                raise ValueError(
                    "Unparseable LLM response (missing true_rows). Raw preview:\n" + (raw or "")[:800]
                )

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
