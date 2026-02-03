# src/portfolio_repo/llm/run1_title_triage_batched.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from portfolio_repo.llm.curnagl_client import TransformersClient


@dataclass(frozen=True)
class Run1Config:
    out_col: str = "title_ai_relevant"

    # how many titles per prompt
    items_per_prompt: int = 40

    # how many prompts per GPU generate() call
    prompts_per_batch: int = 8

    # LLM params
    temperature: float = 0.0
    max_tokens: int = 160

    # resume behavior
    skip_if_already_done: bool = True


_SYSTEM_PROMPT = (
    "Tâche: déterminer si un libellé juridique est en lien direct avec des systèmes automatisés/algorithmique "
    "(y compris traitement automatisé de données, décision automatisée, profilage, logiciels, systèmes informatiques, "
    "infrastructures de calcul, modèles/apprentissage, etc.).\n"
    "Sois CONSERVATEUR: si ce n'est pas clairement impliqué par le libellé, considère que ce n'est PAS pertinent.\n"
    "Tu ne dois répondre qu'avec du JSON strict, sans aucun texte autour."
)


def _make_user_prompt(rows: List[Tuple[int, str]]) -> str:
    lines = "\n".join(f"{uid}\t{label.strip()}" for uid, label in rows)
    return (
        "Décide si chaque libellé implique DIRECTEMENT un système automatisé/algorithmique "
        "(traitement automatisé, décision automatisée, profilage, algorithmes, logiciels/systèmes informatiques, "
        "modèles/apprentissage, infrastructures de calcul). "
        "Sois conservateur.\n"
        "Réponds UNIQUEMENT avec ce JSON strict:\n"
        '{"true_row_uids":[]}\n'
        "en remplaçant [] par les row_uid jugés pertinents.\n"
        "Données (row_uid<TAB>label):\n"
        f"{lines}"
    )


def _parse_true_uids(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # Prefer JSON object extraction
    m = re.search(r"\{[\s\S]*\}", raw)
    cand = m.group(0) if m else raw

    # Normalize common non-JSON booleans/quotes issues (light touch)
    cand = cand.replace("“", '"').replace("”", '"')
    cand = re.sub(r",\s*([}\]])", r"\1", cand)

    try:
        obj = json.loads(cand)
        uids = obj.get("true_row_uids", [])
        if isinstance(uids, list):
            out: List[int] = []
            for x in uids:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out
        return []
    except Exception:
        # Regex fallback: capture integers inside true_row_uids [...]
        m2 = re.search(r"true_row_uids\s*[:=]\s*\[([^\]]*)\]", raw, flags=re.IGNORECASE)
        if not m2:
            return []
        nums = re.findall(r"\d+", m2.group(1))
        return [int(n) for n in nums]


def run1_title_triage_batched(
    client: TransformersClient,
    df: pd.DataFrame,
    cfg: Run1Config,
) -> pd.DataFrame:
    df_out = df.copy()

    if "row_uid" not in df_out.columns:
        df_out["row_uid"] = range(1, len(df_out) + 1)

    if cfg.out_col not in df_out.columns:
        df_out[cfg.out_col] = pd.NA
    elif not cfg.skip_if_already_done:
        df_out[cfg.out_col] = pd.NA

    # We only classify non-articles (same behavior as before: level==5 left untouched)
    df_non = df_out[df_out["level"] != 5].copy()
    if cfg.skip_if_already_done:
        df_non = df_non[df_non[cfg.out_col].isna()].copy()

    if df_non.empty:
        return df_out

    rows: List[Tuple[int, str]] = list(
        zip(df_non["row_uid"].astype(int).tolist(), df_non["label"].astype(str).tolist())
    )

    selected_map: Dict[int, bool] = {}

    # Build user prompts (each prompt contains items_per_prompt rows)
    prompts: List[str] = []
    prompt_chunks: List[List[int]] = []  # row_uids per prompt for fast assignment
    for i in range(0, len(rows), cfg.items_per_prompt):
        chunk = rows[i : i + cfg.items_per_prompt]
        uids = [uid for uid, _ in chunk]
        prompts.append(_make_user_prompt(chunk))
        prompt_chunks.append(uids)

    # Process prompts in GPU batches
    for j in range(0, len(prompts), cfg.prompts_per_batch):
        batch_prompts = prompts[j : j + cfg.prompts_per_batch]
        batch_uids = prompt_chunks[j : j + cfg.prompts_per_batch]

        raws = client.chat_many(
            system_prompt=_SYSTEM_PROMPT,
            user_prompts=batch_prompts,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        for raw, uids in zip(raws, batch_uids):
            true_uids = set(_parse_true_uids(raw))
            allowed = set(uids)
            true_uids = true_uids & allowed

            # Default false for all uids in this prompt
            for uid in uids:
                selected_map[uid] = (uid in true_uids)

            # Hard fail if model returns nothing parseable at all (protect against silent all-false)
            if len(true_uids) == 0 and ("true_row_uids" not in (raw or "")):
                raise ValueError(
                    "Unparseable LLM response (missing true_row_uids). "
                    "Raw preview:\n" + (raw or "")[:800]
                )

    # Write results back
    mask = (df_out["level"] != 5)
    if cfg.skip_if_already_done:
        mask = mask & df_out[cfg.out_col].isna()

    df_out.loc[mask, cfg.out_col] = (
        df_out.loc[mask, "row_uid"].astype(int).map(selected_map).fillna(False).astype(bool)
    )

    return df_out
