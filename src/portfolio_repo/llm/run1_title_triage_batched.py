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
    "Tu reçois une liste de titres juridiques (chapitres/sections).\n"
    "Chaque titre doit être évalué INDIVIDUELLEMENT.\n\n"

    "Important :\n"
    "- Ne suppose AUCUNE relation entre les titres.\n"
    "- Ne te base JAMAIS sur les titres voisins.\n"
    "- Chaque décision doit être prise uniquement à partir du titre concerné.\n\n"

    "Objectif : identifier les titres qui pourraient plausiblement contenir "
    "des dispositions liées aux données, aux systèmes d’information, "
    "aux communications, aux infrastructures techniques ou à leur encadrement.\n\n"

    "Il s’agit d’un TRI LARGE.\n"
    "En cas de doute raisonnable, sélectionne TRUE.\n"
    "Ce filtrage sera affiné ultérieurement.\n\n"

    "Critère :\n"
    "Sélectionne TRUE si le titre évoque, explicitement ou implicitement :\n"
    "- données, traitement, protection, échange, communication,\n"
    "- registres, systèmes d’information, bases de données,\n"
    "- télécommunications, radiocommunication, réseaux,\n"
    "- surveillance technique, dispositifs techniques,\n"
    "- automatisation, infrastructure, interopérabilité,\n"
    "- analyse, statistiques, gestion d’information,\n"
    "- systèmes sectoriels impliquant des données.\n\n"

    "Ne sélectionne FALSE que si le titre est purement générique, "
    "procédural ou sans aucun indice informationnel ou technique.\n\n"

    "Pour CHAQUE titre, fournis une décision et une justification courte.\n\n"

    "Réponds UNIQUEMENT avec ce JSON strict :\n"
    "{\n"
    "  \"decisions\": {\n"
    "    \"row_uid\": {\n"
    "      \"decision\": \"TRUE ou FALSE\",\n"
    "      \"justification\": \"explication courte\"\n"
    "    }\n"
    "  }\n"
    "}\n"
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


def _parse(raw: str) -> Tuple[Dict[int, bool], Dict[int, str]]:
    """
    Returns:
      decisions: {uid: bool}
      justs: {uid: str}  (peut contenir aussi les FALSE si fourni)
    Compatible avec:
      - nouveau format: {"decisions": {"123": {"decision":"TRUE/FALSE","justification":"..."}, ...}}
      - ancien format: {"true_row_uids":[...], "justifications": {...}}
    """
    j = _last_json_obj(raw or "")
    if not j:
        return {}, {}
    try:
        obj = json.loads(j)
    except Exception:
        return {}, {}

    decisions: Dict[int, bool] = {}
    justs: Dict[int, str] = {}

    # Nouveau format
    dmap = obj.get("decisions")
    if isinstance(dmap, dict):
        for k, v in dmap.items():
            try:
                uid = int(k)
            except Exception:
                continue
            if not isinstance(v, dict):
                continue
            dec = str(v.get("decision", "")).strip().upper()
            decisions[uid] = (dec == "TRUE")
            jtxt = v.get("justification", "")
            if isinstance(jtxt, str) and jtxt.strip():
                justs[uid] = jtxt.strip()
        return decisions, justs

    # Ancien format (fallback)
    tuids = set()
    for x in obj.get("true_row_uids", []):
        try:
            tuids.add(int(x))
        except Exception:
            pass
    for uid in tuids:
        decisions[uid] = True

    jmap = obj.get("justifications", {})
    if isinstance(jmap, dict):
        for k, v in jmap.items():
            try:
                uid = int(k)
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                justs[uid] = v.strip()

    return decisions, justs


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

            tdec, tjust = _parse(raw)
            if not tdec and not tjust:
                raw2 = client.chat_many(
                    _SYSTEM_PROMPT + "\n\nRAPPEL CRITIQUE: réponds uniquement avec le JSON, sans répéter le prompt.",
                    [_make_user_prompt(ch)],
                    temperature=0.0,
                    max_tokens=cfg.max_tokens,
                )[0]
                tdec, tjust = _parse(raw2)

            if not tdec and not tjust:
                bad += 1

            for u in uids:
                # défaut = FALSE si absent du JSON
                sel[u] = bool(tdec.get(u, False))
                # justification pour TRUE ET FALSE (si le modèle la donne)
                jtxt = tjust.get(u)
                jus[u] = jtxt if isinstance(jtxt, str) and jtxt.strip() else pd.NA



    mask = out["level"].isin([1, 2, 3, 4])
    if cfg.skip_if_already_done:
        mask = mask & out[cfg.out_col].isna()

    u = out.loc[mask, "row_uid"].astype(int)
    out.loc[mask, cfg.out_col] = u.map(sel).fillna(False).astype(bool)
    out.loc[mask, cfg.out_just_col] = [jus.get(int(x), pd.NA) for x in u.tolist()]

    if bad:
        print(f"[RUN1] WARN: {bad} prompt(s) non-parseables après retry (forcés FALSE).")

    return out
