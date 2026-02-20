from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


SYSTEM_PROMPT_FR = (
    "Tu es un système STRICT de classification juridique.\n"
    "Tu dois répondre exactement sur DEUX lignes, dans cet ordre, sans texte supplémentaire:\n"
    "RELEVANT: TRUE ou FALSE\n"
    "JUSTIFICATION: une justification courte (1-2 phrases) en français\n"
    "Ne mets rien d'autre."
)

USER_TEMPLATE_FR = """Décide si cet article fixe des règles/obligations/procédures/responsabilités/exigences
qui influence l’implémentation, le développement, la sécurité, l’usage ou la responsabilité
d’une intelligence artificielle (au sens large) : système qui utilise des algorithmes et des données pour analyser, prédire et automatiser des tâches ou des décisions.

Important:
- TRUE seulement s’il y a un effet normatif concret lié à ces systèmes (conditions, exigences, surveillance, obligations, responsabilités, procédure, autorisations, etc.).
- FALSE si c’est seulement descriptif/définition générale ou sans portée normative liée à ces systèmes.

Texte légal:
{article_text}
"""


def build_user_prompt(article_text: str) -> str:
    return USER_TEMPLATE_FR.format(article_text=(article_text or "").strip())


def parse_relevant_justif(raw: str) -> Tuple[Optional[bool], Optional[str], bool]:
    """
    Robust parse: take the LAST match in the string.
    Accepts variants with extra spaces. Returns (relevant, justification, ok).
    """
    if raw is None:
        return None, None, False

    s = str(raw)

    # find all occurrences; take last to avoid matching "TRUE or FALSE" in the prompt if present
    rel_matches = re.findall(r"RELEVANT:\s*(TRUE|FALSE)\b", s, flags=re.IGNORECASE)
    jus_matches = re.findall(r"JUSTIFICATION:\s*(.+)", s, flags=re.IGNORECASE)

    if not rel_matches or not jus_matches:
        return None, None, False

    rel_token = rel_matches[-1].upper()
    jus = jus_matches[-1].strip()

    if rel_token not in ("TRUE", "FALSE"):
        return None, None, False
    if not jus:
        return None, None, False

    relevant = True if rel_token == "TRUE" else False
    jus_clean = " ".join(jus.split())
    return relevant, jus_clean, True


@dataclass(frozen=True)
class ApertusConfig:
    model_path: str
    dtype: str = "bf16"
    trust_remote_code: bool = True
    temperature: float = 0.0
    max_tokens: int = 220  # need room for justification


class ApertusBatchClassifier:
    def __init__(self, cfg: ApertusConfig):
        from portfolio_repo.llm.curnagl_client import TransformersClient, TransformersConfig  # type: ignore

        self.cfg = cfg
        self.client = TransformersClient(
            TransformersConfig(
                model_path=cfg.model_path,
                dtype=cfg.dtype,
                trust_remote_code=cfg.trust_remote_code,
            )
        )

    def classify_batch_raw(
        self, article_texts: List[str]
    ) -> Tuple[List[Optional[bool]], List[Optional[str]], List[str], List[bool]]:
        user_prompts = [build_user_prompt(t) for t in article_texts]
        outs = self.client.chat_many(
            system_prompt=SYSTEM_PROMPT_FR,
            user_prompts=user_prompts,
            temperature=float(self.cfg.temperature),
            max_tokens=int(self.cfg.max_tokens),
        )

        raws = ["" if o is None else str(o) for o in outs]

        relevants: List[Optional[bool]] = []
        justifs: List[Optional[str]] = []
        oks: List[bool] = []

        for raw in raws:
            rel, jus, ok = parse_relevant_justif(raw)
            relevants.append(rel)
            justifs.append(jus)
            oks.append(ok)

        return relevants, justifs, raws, oks