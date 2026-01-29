# src/portfolio_repo/llm/article_run2_allinone.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from portfolio_repo.llm.client import LocalLLMClient


@dataclass(frozen=True)
class Run2AllInOneConfig:
    # LLM
    temperature: float = 0.0
    max_tokens: int = 900

    # selection + outputs
    out_selected_col: str = "run2_selected"
    out_ai_relevant_col: str = "article_ai_relevant"
    out_targets_col: str = "article_targets"  # JSON string list
    out_blocks_col: str = "article_target_blocks"  # JSON string list[dict]
    out_justification_col: str = "article_justification"

    skip_if_already_done: bool = True

    # debug
    debug_prompt: bool = False
    debug_raw_response: bool = False
    debug_max_chars: int = 6000


# Instrument taxonomy (short + discriminant)
_ALLOWED_INSTRUMENTS = [
    "REGLE_OBLIGATION_INTERDICTION",
    "AUTORISATION_LICENCE_APPROBATION",
    "SURVEILLANCE_AUDIT_CONTROLE",
    "EVALUATION_RISQUES_SECURITE",
    "TRANSPARENCE_INFORMATION_REPORTING",
    "NORMES_CERTIFICATION_CONFORMITE",
    "GOUVERNANCE_AUTORITE_RECOURS",
    "INCITATIONS_MARCHE_MARCHES_PUBLICS",
    "CAPACITE_FORMATION",
    "EXPERIMENTATION_SANDBOX",
    "NONE",
]

_ALLOWED_TARGETS = {"Data", "Computing Infrastructure", "Development & Adoption", "Skills"}


_SYSTEM_PROMPT = (
    "Tu es un codeur juridique strict.\n"
    "Tâche: coder UN article.\n\n"
    "1) Détecter si le texte mentionne un système automatisé/algorithmique (ex: traitement automatisé, "
    "décision automatisée, profilage, algorithme, système informatique, logiciel, journalisation/log, "
    "interface, liaison de communication, modèle/apprentissage).\n"
    "2) Décider ai_relevant:\n"
    "- si aucun système automatisé -> ai_relevant=false.\n"
    "- sinon ai_relevant=true seulement si l'article fixe des conditions/obligations/autorisation/"
    "contrôle/exigences techniques/transparence liées à ce système (directement ou via l'objet régulé).\n"
    "3) Si ai_relevant=true: attribuer au moins une target parmi:\n"
    "- Development & Adoption (usage/déploiement/supervision/responsabilité)\n"
    "- Data (données: collecte, accès, logs, transmission, effacement)\n"
    "- Computing Infrastructure (sécurité, accès, stockage, intégrité, communication)\n"
    "- Skills (formation/qualification/ressources)\n"
    "4) Pour chaque target: choisir exactement un instrument (liste fournie) + 1–3 indices textuels courts.\n\n"
    "Sortie: JSON strict uniquement."
)


_USER_PROMPT_TEMPLATE = """Analyse cet article et retourne uniquement du JSON.

row_uid: {row_uid}

TEXTE:
\"\"\"{article_text}\"\"\"

VALEURS AUTORISÉES:
- targets: Development & Adoption | Data | Computing Infrastructure | Skills
- instruments: {allowed_instruments}

FORMAT JSON EXACT:
{{
  "row_uid": {row_uid},
  "automation_present": <true|false>,
  "automation_trigger": "<extrait exact ou vide>",
  "ai_relevant": <true|false>,
  "blocks": [
    {{
      "target": "<une target autorisée>",
      "instrument": "<une valeur autorisée>",
      "evidence_terms": ["<1..3 extraits exacts courts>"],
      "justification": "<1 phrase>"
    }}
  ],
  "global_justification": "<1-2 phrases>"
}}

CONTRAINTES:
- Si ai_relevant=false -> blocks=[]
- Si ai_relevant=true -> blocks longueur >= 1
- evidence_terms: extraits courts (pas de longues citations)
"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_BOOL_FIXES = [
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
]


def _try_repair_json(text: str) -> str:
    t = text.strip()
    m = _JSON_BLOCK_RE.search(t)
    if m:
        t = m.group(0).strip()
    for rx, repl in _BOOL_FIXES:
        t = rx.sub(repl, t)
    t = _TRAILING_COMMA_RE.sub(r"\1", t)
    if '"' not in t and "'" in t:
        t = t.replace("'", '"')
    return t


def _safe_json_loads(raw: str) -> dict:
    return json.loads(_try_repair_json(raw))


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["law_id", "order_index", "level", "node_id"] if c in df.columns]
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def compute_run2_selection(
    df: pd.DataFrame,
    title_relevant_col: str = "title_ai_relevant",
    out_selected_col: str = "run2_selected",
) -> pd.DataFrame:
    """
    Mark articles (level=5) that are under an "active" relevant title (levels 0..4).
    """
    if title_relevant_col not in df.columns:
        raise ValueError(f"Missing column {title_relevant_col}. Run 1 output required.")

    df2 = _stable_sort(df).copy()
    df2[out_selected_col] = pd.NA

    for _, grp_idx in df2.groupby("law_id", sort=False).groups.items():
        idx_list = list(grp_idx)
        active: List[Optional[bool]] = [None, None, None, None, None]  # levels 0..4

        for i in idx_list:
            lvl = int(df2.at[i, "level"])
            if lvl < 5:
                val = df2.at[i, title_relevant_col]
                b = bool(val) if pd.notna(val) else False
                active[lvl] = b
                for d in range(lvl + 1, 5):
                    active[d] = None
                df2.at[i, out_selected_col] = pd.NA
            else:
                df2.at[i, out_selected_col] = any(v is True for v in active)

    return df2


def classify_selected_articles_allinone(
    client: LocalLLMClient,
    df: pd.DataFrame,
    cfg: Run2AllInOneConfig,
    title_relevant_col: str = "title_ai_relevant",
) -> pd.DataFrame:
    """
    Adds for selected articles:
    - ai_relevant (bool)
    - targets (json list[str])
    - blocks (json list[dict] with target+instrument)
    - justification (str)
    """
    df2 = compute_run2_selection(
        df=df,
        title_relevant_col=title_relevant_col,
        out_selected_col=cfg.out_selected_col,
    ).copy()

    for col in [cfg.out_ai_relevant_col, cfg.out_targets_col, cfg.out_blocks_col, cfg.out_justification_col]:
        if col not in df2.columns:
            df2[col] = pd.NA

    df2 = _stable_sort(df2)

    for _, df_law in df2.groupby("law_id", sort=False):
        df_law = df_law.reset_index()  # original index stored in "index"

        for pos in range(len(df_law)):
            row = df_law.iloc[pos]
            if int(row["level"]) != 5:
                continue

            if row[cfg.out_selected_col] is not True:
                continue

            original_idx = int(row["index"])

            if cfg.skip_if_already_done and pd.notna(df2.at[original_idx, cfg.out_ai_relevant_col]):
                continue

            article_text = str(row.get("text", "") or "").strip()
            if not article_text:
                df2.at[original_idx, cfg.out_ai_relevant_col] = False
                df2.at[original_idx, cfg.out_targets_col] = json.dumps([], ensure_ascii=False)
                df2.at[original_idx, cfg.out_blocks_col] = json.dumps([], ensure_ascii=False)
                df2.at[original_idx, cfg.out_justification_col] = "Texte de l'article vide."
                continue

            row_uid = int(row.get("row_uid"))

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_PROMPT_TEMPLATE.format(
                        row_uid=row_uid,
                        article_text=article_text,
                        allowed_instruments=", ".join(_ALLOWED_INSTRUMENTS),
                    ),
                },
            ]

            if cfg.debug_prompt:
                print("\n" + "=" * 90)
                print(f"[RUN2-ALL DEBUG] row_uid={row_uid} node_id={row.get('node_id')}")
                print("-" * 90)
                for m in messages:
                    role = m.get("role", "")
                    content = str(m.get("content", "") or "")
                    if len(content) > cfg.debug_max_chars:
                        content = content[: cfg.debug_max_chars] + "\n... [TRUNCATED]"
                    print(f"\n[{role.upper()}]\n{content}")
                print("=" * 90 + "\n")

            raw = client.chat(
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                response_format={"type": "json_object"},
            )

            if cfg.debug_raw_response:
                preview = raw if len(raw) <= cfg.debug_max_chars else raw[: cfg.debug_max_chars] + "\n... [TRUNCATED]"
                print("\n" + "-" * 90)
                print("[RUN2-ALL DEBUG] RAW MODEL OUTPUT")
                print(preview)
                print("-" * 90 + "\n")

            data = _safe_json_loads(raw)

            ai_rel = bool(data.get("ai_relevant", False))
            blocks = data.get("blocks", [])
            if not isinstance(blocks, list):
                blocks = []

            # Normalize + validate blocks
            norm_blocks = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                target = b.get("target")
                instrument = b.get("instrument")
                if not isinstance(target, str) or target not in _ALLOWED_TARGETS:
                    continue
                if not isinstance(instrument, str) or instrument not in _ALLOWED_INSTRUMENTS:
                    instrument = "NONE"

                ev = b.get("evidence_terms", [])
                if not isinstance(ev, list):
                    ev = []
                ev = [str(x) for x in ev if isinstance(x, str) and str(x).strip()][:3]

                just = str(b.get("justification", "") or "").strip()
                if not just:
                    just = "Aucune justification."

                norm_blocks.append(
                    {
                        "target": target,
                        "instrument": instrument,
                        "evidence_terms": ev,
                        "justification": just,
                    }
                )

            if not ai_rel:
                norm_blocks = []
            else:
                # ai_relevant=true => must have >=1 block; if model failed, force safe fallback
                if len(norm_blocks) == 0:
                    # conservative fallback: keep ai_rel=true but record NONE to avoid empty
                    norm_blocks = [
                        {
                            "target": "Development & Adoption",
                            "instrument": "NONE",
                            "evidence_terms": [],
                            "justification": "ai_relevant=true mais instrument/target non identifiés de manière fiable.",
                        }
                    ]

            targets = [b["target"] for b in norm_blocks]
            global_justif = str(data.get("global_justification", "") or "").strip()
            if not global_justif:
                global_justif = "Aucune justification fournie."

            df2.at[original_idx, cfg.out_ai_relevant_col] = ai_rel
            df2.at[original_idx, cfg.out_targets_col] = json.dumps(targets, ensure_ascii=False)
            df2.at[original_idx, cfg.out_blocks_col] = json.dumps(norm_blocks, ensure_ascii=False)
            df2.at[original_idx, cfg.out_justification_col] = global_justif

    return df2
