# src/runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    id_col: str
    text_col: str
    batch_size: int = 32
    temperature: float = 0.0
    max_new_tokens: int = 200


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def run_llm_dataframe(
    df: pd.DataFrame,
    cfg: RunConfig,
    client,
    system_prompt: str,
    select_mask_fn: Callable[[pd.DataFrame], pd.Series],
    build_prompt_fn: Callable[[pd.Series, str], str],
    parse_fn: Callable[[str], Dict[str, object]],
    output_cols: List[str],
    skip_if_already_filled: Optional[str] = None,
    required_cols: Optional[List[str]] = None,
    max_retries: int = 2,
) -> pd.DataFrame:
    """
    - select_mask_fn(df) -> bool mask des lignes à traiter
    - build_prompt_fn(row, text_col) -> string prompt user
    - parse_fn(raw_completion) -> dict {col: value, ...}
    - output_cols = colonnes qu'on garantit dans df
    - skip_if_already_filled: si défini, on saute les lignes où df[col] n'est pas NA
    - required_cols: colonnes qui doivent être non-NA pour considérer une ligne comme complète ;
      si une ligne reste incomplète après un appel, elle est retentée (jusqu'à max_retries fois de plus)
      avant d'être abandonnée. Par défaut, égal à output_cols.
    - max_retries: nombre de tentatives supplémentaires pour les lignes dont la réponse est incomplète.
    """
    df = df.copy()
    df = ensure_columns(df, output_cols)

    mask = select_mask_fn(df)
    if mask is None:
        mask = pd.Series([True] * len(df), index=df.index)

    todo = mask.copy()
    if skip_if_already_filled:
        todo = todo & df[skip_if_already_filled].isna()

    idx = df.index[todo].tolist()
    if not idx:
        return df

    required = required_cols or output_cols

    pending = idx
    attempt = 0
    while pending and attempt <= max_retries:
        next_pending: List = []

        # batching
        for start in range(0, len(pending), int(cfg.batch_size)):
            batch_idx = pending[start:start + int(cfg.batch_size)]
            batch_rows = df.loc[batch_idx]

            user_prompts = [build_prompt_fn(row, cfg.text_col) for _, row in batch_rows.iterrows()]
            raw = client.chat_many(
                system_prompt=system_prompt,
                user_prompts=user_prompts,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
            )

            # write back
            for i, row_id in enumerate(batch_idx):
                parsed = parse_fn(raw[i])
                for k, v in parsed.items():
                    if k in df.columns:
                        df.at[row_id, k] = v

                if any(pd.isna(df.at[row_id, c]) for c in required):
                    next_pending.append(row_id)

        if next_pending:
            print(f"[RETRY] {len(next_pending)} ligne(s) incomplète(s) après tentative {attempt + 1}, "
                  f"nouvelle tentative...")

        pending = next_pending
        attempt += 1

    return df