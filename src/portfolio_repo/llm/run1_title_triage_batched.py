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
    "Tu es un juriste-analyste spécialisé dans l’identification des dispositifs juridiques "
    "liés à la gestion, au traitement ou à l’exploitation structurée de l’information.\n\n"

    "Tu reçois ci-dessous une liste de titres, sous-titres, chapitres ou sections "
    "représentant la structure des textes de lois suisses.\n"
    "Tu n'as accès qu'aux TITRES / INTITULÉS, pas au texte des articles.\n\n"

    "Tâche : identifier les titres qui annoncent un contenu normatif portant sur un "
    "DISPOSITIF INFORMATIONNEL CONCRET, c’est-à-dire notamment :\n"
    "- un système d’information, un registre ou une base de données,\n"
    "- un traitement structuré de données (personnelles ou non),\n"
    "- une surveillance technique ou une communication électronique,\n"
    "- une interconnexion de systèmes ou une infrastructure informationnelle,\n"
    "- ou plus généralement l’utilisation opérationnelle de technologies informationnelles.\n\n"

    "Règle fondamentale :\n"
    "Sélectionne un titre uniquement s’il est raisonnable de s’attendre, sur la base de son libellé, "
    "à des règles MATERIELLES concernant la collecte, le traitement, la transmission, "
    "l’interconnexion ou l’exploitation de données ou de systèmes informationnels.\n\n"

    "Ne PAS sélectionner les titres qui sont :\n"
    "- purement structurels ou introductifs (ex. Dispositions générales, Objet et champ d’application, "
    "Principes, Définitions),\n"
    "- exclusivement procéduraux ou institutionnels (ex. Procédure, Compétence, Voies de droit, "
    "Organisation des autorités),\n"
    "- ou qui n’annoncent aucun dispositif informationnel identifiable, même s’ils appartiennent "
    "à un domaine technique ou réglementé.\n\n"

    "Important :\n"
    "- Tu peux t’appuyer sur ta compréhension générale du droit et des politiques publiques.\n"
    "- N’infère pas le contenu précis des articles : raisonne uniquement à partir du titre.\n"
    "- Privilégie une interprétation substantielle (objet matériel du titre), pas formelle.\n"
    "- Évite toute sélection par simple prudence : sélectionne uniquement lorsqu’un "
    "dispositif informationnel est clairement suggéré par le titre.\n\n"

    "Réponds UNIQUEMENT avec ce JSON strict :\n"
    "{\n"
    "  \"true_row_uids\": [],\n"
    "  \"justifications\": {}\n"
    "}\n"
    "en remplaçant [] par les row_uid jugés pertinents.\n"
    "Pour chaque row_uid sélectionné, ajoute une justification courte dans \"justifications\" "
    "(clé = row_uid, valeur = justification).\n"
    "Si aucun titre ne doit être sélectionné, retourne des listes vides.\n"
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
