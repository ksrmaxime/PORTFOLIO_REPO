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
    "Tu reçois ci-dessous une liste des titres, sous-titres, chapitres etc... représentant la structure des textes de lois suisses.\n"
    "Tu n'as accès qu'aux TITRES/INTITULÉS, pas au texte des articles qui composent leur section.\n\n"
    "Ton rôle est de déterminer si certains de ces titres sont à la tête d'une section qui contient des articles étant en lien avec de l'intelligence artificielle.\n"
    "Nous définissions l'intelligence artificielle de manière large : cela inclut les systèmes automatisés/algorithmes, le traitement automatisé de données, les infrastructures informatiques/de calcul (serveurs, cloud), ainsi que l'intelligence artificielle au sens strict.\n"
    "Afin de savoir si un titre est pertinent, il faut se demander si les articles qui suivront légifèrent sur le développement, l'utilisation, la régulation ou les implications de tels systèmes.\n\n"
    "A ce stade de l'analyse, l'onjectif est de pouvoir élimner les sections qui ne sont pas en lien avec ce type de système.\n"
    "Il est donc important de ne conserver que les titres qui font explicitement référence à notre définition. Base toi sur les mots du titre et n'essaye pas d'intrepréter.\n\n"
    "Il est donc prévisible qu'une grosse partie des titres ne soient pas en lien.\n\n"
    "Réponds UNIQUEMENT avec ce JSON strict:\n"
    '{"true_row_uids":[]}\n'
    "en remplaçant [] par les row_uid jugés pertinents.\n"
    "Tu ne dois répondre qu'avec du JSON strict, sans aucun texte autour."
)


def _make_user_prompt(rows: List[Tuple[int, str]]) -> str:
    lines = "\n".join(f"{uid}\t{label.strip()}" for uid, label in rows)
    return (
        "Données (row_uid<TAB>label):\n"
        f"{lines}"
    )


def _parse_true_uids(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # 1) Trouver tous les objets JSON qui contiennent "true_row_uids" et prendre le dernier
    candidates = re.findall(r"\{[\s\S]*?\"true_row_uids\"[\s\S]*?\}", raw)
    cand = candidates[-1] if candidates else raw

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
        # fallback regex: prendre la dernière occurrence de true_row_uids [...]
        m2_all = re.findall(r"true_row_uids\s*[:=]\s*\[([^\]]*)\]", raw, flags=re.IGNORECASE)
        if not m2_all:
            return []
        nums = re.findall(r"\d+", m2_all[-1])
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
    df_non = df_out[df_out["level"].isin([1, 2, 3, 4])].copy()
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

        for raw, uids in zip(raws, batch_uids):
            true_uids = set(_parse_true_uids(raw))
            allowed = set(uids)
            true_uids = true_uids & allowed

            # Default false for all uids in this prompt
            for uid in uids:
                selected_map[uid] = (uid in true_uids)

            # Optionnel: garde-fou contre sorties totalement non parseables
            if len(true_uids) == 0 and ("true_row_uids" not in (raw or "")):
                raise ValueError(
                    "Unparseable LLM response (missing true_row_uids). Raw preview:\n"
                    + (raw or "")[:800]
                )


    # Write results back
    mask = df_out["level"].isin([1, 2, 3, 4])
    if cfg.skip_if_already_done:
        mask = mask & df_out[cfg.out_col].isna()

    df_out.loc[mask, cfg.out_col] = (
        df_out.loc[mask, "row_uid"].astype(int).map(selected_map).fillna(False).astype(bool)
    )

    return df_out
