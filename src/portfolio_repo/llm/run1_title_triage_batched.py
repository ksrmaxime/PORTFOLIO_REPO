from __future__ import annotations

import json, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from portfolio_repo.llm.curnagl_client import TransformersClient

from pathlib import Path
import json

_RAW_DIR = Path("logs/run1_raw_responses")
_RAW_DIR.mkdir(parents=True, exist_ok=True)

def _dump_raw(prompt_index: int, uids: list[int], raw: str, tag: str = "main") -> None:
    """
    Écrit la réponse brute du LLM + les uids associés pour debug.
    """
    rec = {
        "prompt_index": prompt_index,
        "tag": tag,
        "uids": uids,
        "raw": raw,
    }
    (_RAW_DIR / f"raw_{prompt_index:05d}_{tag}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )



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
    "Tu reçois une liste de titres juridiques (chapitres/sections) issus de textes de lois.\n"
    "Tu n’as accès qu’aux TITRES, pas aux articles.\n\n"

    "IMPORTANT : chaque titre doit être évalué INDIVIDUELLEMENT.\n"
    "- Ne suppose AUCUNE relation entre les titres.\n"
    "- Ne te base JAMAIS sur les titres voisins.\n"
    "- La décision doit dépendre uniquement du titre concerné.\n\n"

    "Objectif : TRI LARGE (haute sensibilité).\n"
    "Marque TRUE si le titre pourrait plausiblement contenir des dispositions liées à :\n"
    "- données/information (collecte, traitement, protection, accès, échange, transmission),\n"
    "- systèmes d’information / registres / bases de données,\n"
    "- communication électronique / télécommunications / radiocommunication / réseaux,\n"
    "- cybersécurité / cybermenaces / sécurité informatique,\n"
    "- automatisation / interopérabilité / infrastructures numériques,\n"
    "- surveillance TECHNIQUE (vidéosurveillance, surveillance des télécoms, dispositifs techniques).\n\n"

    "En cas de doute raisonnable, sélectionne TRUE (ce filtrage sera affiné ensuite).\n\n"

    "Contraintes strictes :\n"
    "- Le JSON DOIT contenir une entrée dans \"decisions\" pour CHAQUE row_uid fourni.\n"
    "- \"justification\" ne doit JAMAIS être vide.\n"
    "- Si decision = \"FALSE\", la justification doit expliquer brièvement l’absence d’indice "
    "(ex. \"Aucun indice informationnel/technique explicite dans le titre\").\n\n"

    "Réponds UNIQUEMENT avec ce JSON strict (une décision et une justification pour CHAQUE row_uid) :\n"
    "{\n"
    "  \"decisions\": {\n"
    "    \"row_uid\": {\n"
    "      \"decision\": \"TRUE\" ,\n"
    "      \"justification\": \"...\"\n"
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


from typing import Dict, Tuple

def _parse(raw: str) -> Tuple[Dict[int, bool], Dict[int, str]]:
    """
    Retourne:
      - decisions: {uid: bool}
      - justs: {uid: str}
    Supporte:
      - Nouveau format: {"decisions": {"123": {"decision":"TRUE/FALSE","justification":"..."}, ...}}
      - Ancien format: {"true_row_uids":[...], "justifications": {...}}
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
            if isinstance(jtxt, str):
                jtxt = jtxt.strip()
                if jtxt:
                    justs[uid] = jtxt
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
            if isinstance(v, str):
                v = v.strip()
                if v:
                    justs[uid] = v

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

        # Exemple: tu es déjà dans un loop sur des chunks/batches
# et tu as uids = list[int] correspondant aux titres du chunk,
# plus raw = réponse LLM.

raw = client.chat_many(
    _SYSTEM_PROMPT,
    [_make_user_prompt(ch)],
    temperature=0.0,
    max_tokens=cfg.max_tokens,
)[0]

_dump_raw(prompt_index=i, uids=[int(x) for x in uids], raw=raw, tag="main")

tdec, tjust = _parse(raw)

# Retry si parsing vide (JSON cassé / tronqué)
if not tdec and not tjust:
    raw2 = client.chat_many(
        _SYSTEM_PROMPT + "\n\nRAPPEL CRITIQUE: réponds uniquement avec le JSON, sans répétition.",
        [_make_user_prompt(ch)],
        temperature=0.0,
        max_tokens=cfg.max_tokens,
    )[0]
    _dump_raw(prompt_index=i, uids=[int(x) for x in uids], raw=raw2, tag="retry")
    tdec, tjust = _parse(raw2)

# Remplissage final (garantit une justification non vide pour chaque uid)
for u in uids:
    decision = bool(tdec.get(u, False))  # défaut FALSE si absent
    sel[u] = decision

    jtxt = tjust.get(u)
    if isinstance(jtxt, str):
        jtxt = jtxt.strip()

    if not jtxt:
        # Justification par défaut (utile si le modèle omet les FALSE)
        if decision:
            jtxt = "Indication plausible d’un enjeu informationnel/technique dans le titre."
        else:
            jtxt = "Aucun indice informationnel/technique explicite dans le titre."

    jus[u] = jtxt




    mask = out["level"].isin([1, 2, 3, 4])
    if cfg.skip_if_already_done:
        mask = mask & out[cfg.out_col].isna()

    u = out.loc[mask, "row_uid"].astype(int)
    out.loc[mask, cfg.out_col] = u.map(sel).fillna(False).astype(bool)
    out.loc[mask, cfg.out_just_col] = [jus.get(int(x), pd.NA) for x in u.tolist()]

    if bad:
        print(f"[RUN1] WARN: {bad} prompt(s) non-parseables après retry (forcés FALSE).")

    return out
