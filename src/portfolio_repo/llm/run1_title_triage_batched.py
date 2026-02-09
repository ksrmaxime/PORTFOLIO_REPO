from __future__ import annotations

import json, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from portfolio_repo.llm.curnagl_client import TransformersClient


@dataclass(frozen=True)
class Run1Config:
    out_col: str = "title_ai_relevant"
    out_just_col: str = "title_ai_justification"
    items_per_prompt: int = 40
    prompts_per_batch: int = 8
    temperature: float = 0.0
    max_tokens: int = 220
    skip_if_already_done: bool = True


# PROMPT: keep EXACTLY as in your current file (do not edit wording)
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
    return "Données (row_uid<TAB>label):\n" + "\n".join(f"{u}\t{t.strip()}" for u, t in rows)


def _last_json_obj(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip().replace("“", '"').replace("”", '"')
    s = re.sub(r",\s*([}\]])", r"\1", s)
    depth = 0
    start = None
    last = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                last = s[start : i + 1]
                start = None
    return last


def _parse(raw: str) -> Tuple[List[int], Dict[int, str]]:
    j = _last_json_obj(raw or "")
    if not j:
        return [], {}
    try:
        obj = json.loads(j)
    except Exception:
        return [], {}

    uids: List[int] = []
    for x in obj.get("true_row_uids", []):
        try:
            uids.append(int(x))
        except Exception:
            pass

    justs: Dict[int, str] = {}
    jmap = obj.get("justifications", {})
    if isinstance(jmap, dict):
        for k, v in jmap.items():
            try:
                kk = int(k)
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                justs[kk] = v.strip()

    return uids, justs


def run1_title_triage_batched(client: TransformersClient, df: pd.DataFrame, cfg: Run1Config) -> pd.DataFrame:
    out = df.copy()
    if "row_uid" not in out.columns:
        out["row_uid"] = range(1, len(out) + 1)

    for c in (cfg.out_col, cfg.out_just_col):
        if c not in out.columns or not cfg.skip_if_already_done:
            out[c] = pd.NA

    # only level 1..4
    cand = out[out["level"].isin([1, 2, 3, 4])].copy()
    if cfg.skip_if_already_done:
        cand = cand[cand[cfg.out_col].isna()].copy()
    if cand.empty:
        return out

    rows = list(zip(cand["row_uid"].astype(int).tolist(), cand["label"].astype(str).tolist()))
    prompts, chunks = [], []
    for i in range(0, len(rows), cfg.items_per_prompt):
        ch = rows[i : i + cfg.items_per_prompt]
        prompts.append(_make_user_prompt(ch))
        chunks.append(ch)

    sel: Dict[int, bool] = {}
    jus: Dict[int, object] = {}
    bad = 0

    for j in range(0, len(prompts), cfg.prompts_per_batch):
        bprompts = prompts[j : j + cfg.prompts_per_batch]
        bchunks = chunks[j : j + cfg.prompts_per_batch]

        raws = client.chat_many(_SYSTEM_PROMPT, bprompts, temperature=cfg.temperature, max_tokens=cfg.max_tokens)

        for raw, ch in zip(raws, bchunks):
            uids = [u for u, _ in ch]
            allowed = set(uids)

            tuids, tjust = _parse(raw)
            if not tuids and not tjust:
                # retry once (prompt echo / garbage)
                raw2 = client.chat_many(
                    _SYSTEM_PROMPT + "\n\nRAPPEL CRITIQUE: réponds uniquement avec le JSON, sans répéter le prompt.",
                    [_make_user_prompt(ch)],
                    temperature=0.0,
                    max_tokens=cfg.max_tokens,
                )[0]
                tuids, tjust = _parse(raw2)

            if not tuids and not tjust:
                bad += 1

            tset = set(tuids) & allowed
            for u in uids:
                sel[u] = (u in tset)
                if u in tset:
                    jtxt = tjust.get(u)
                    jus[u] = jtxt.strip() if isinstance(jtxt, str) and jtxt.strip() else pd.NA
                else:
                    jus[u] = pd.NA


    mask = out["level"].isin([1, 2, 3, 4])
    if cfg.skip_if_already_done:
        mask = mask & out[cfg.out_col].isna()

    u = out.loc[mask, "row_uid"].astype(int)
    out.loc[mask, cfg.out_col] = u.map(sel).fillna(False).astype(bool)
    out.loc[mask, cfg.out_just_col] = [jus.get(int(x), pd.NA) for x in u.tolist()]

    if bad:
        print(f"[RUN1] WARN: {bad} prompt(s) non-parseables après retry (forcés FALSE).")

    return out
