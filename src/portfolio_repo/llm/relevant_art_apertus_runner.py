from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


SYSTEM_PROMPT = (
    "You are a strict legal classification system. "
    "Return ONLY one token: TRUE or FALSE. "
    "No explanation, no punctuation, no extra words."
)

USER_TEMPLATE = """Does the following Swiss legal provision establish rules that can affect the implementation, development, security, use, accountability or responsibility regarding automated systems, registers, software, automated infrastructure or AI-related systems (broad sense of AI)?

Return ONLY TRUE or FALSE.

LEGAL TEXT:
{article_text}
"""


def build_user_prompt(article_text: str) -> str:
    return USER_TEMPLATE.format(article_text=(article_text or "").strip())


def parse_true_false_strict(raw: str) -> Optional[bool]:
    """
    Strict parsing:
    - Accepts ONLY 'TRUE' or 'FALSE' as the full answer (ignoring surrounding whitespace).
    - If model returns anything else (e.g., 'TRUE.' or 'TRUE because...'), returns None.
    """
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if s == "TRUE":
        return True
    if s == "FALSE":
        return False
    return None


@dataclass(frozen=True)
class ApertusConfig:
    model_path: str
    dtype: str = "bf16"          # "bf16" | "fp16"
    trust_remote_code: bool = True
    temperature: float = 0.0
    max_tokens: int = 32         # max_new_tokens côté generate()


class ApertusBatchClassifier:
    """
    Wrapper around src/portfolio_repo/llm/curnagl_client.py::TransformersClient
    Uses chat_many() for true GPU batching.
    """

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

    def classify_batch_raw(self, article_texts: List[str]) -> Tuple[List[Optional[bool]], List[str]]:
        user_prompts = [build_user_prompt(t) for t in article_texts]
        outs = self.client.chat_many(
            system_prompt=SYSTEM_PROMPT,
            user_prompts=user_prompts,
            temperature=float(self.cfg.temperature),
            max_tokens=int(self.cfg.max_tokens),
        )
        # Ensure string outputs
        outs_str = ["" if o is None else str(o) for o in outs]
        preds = [parse_true_false_strict(o) for o in outs_str]
        return preds, outs_str