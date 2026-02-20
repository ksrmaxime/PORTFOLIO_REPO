from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


SYSTEM_PROMPT = (
    "You are a strict legal classification system. "
    "Return ONLY one token: TRUE or FALSE."
)

USER_TEMPLATE = """Does the following Swiss legal provision establish rules that can affect the implementation, development, security, use, accountability or responsibility regarding automated systems, registers, software, automated infrastructure or AI-related systems (broad sense of AI)?

LEGAL TEXT:
{article_text}
"""


def build_user_prompt(article_text: str) -> str:
    return USER_TEMPLATE.format(article_text=article_text.strip())


def parse_true_false(raw: str) -> Optional[bool]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    m = re.search(r"\b(TRUE|FALSE)\b", s)
    if not m:
        return None
    return True if m.group(1) == "TRUE" else False


@dataclass(frozen=True)
class ApertusConfig:
    model_path: str
    dtype: str = "bf16"          # "bf16" | "fp16"
    trust_remote_code: bool = True
    temperature: float = 0.0
    max_tokens: int = 32         # max_new_tokens côté generate()


class ApertusBatchClassifier:
    """
    Thin wrapper around src/portfolio_repo/llm/curnagl_client.py::TransformersClient
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

    def classify_batch(self, article_texts: List[str]) -> List[Optional[bool]]:
        user_prompts = [build_user_prompt(t or "") for t in article_texts]
        outs = self.client.chat_many(
            system_prompt=SYSTEM_PROMPT,
            user_prompts=user_prompts,
            temperature=float(self.cfg.temperature),
            max_tokens=int(self.cfg.max_tokens),
        )
        return [parse_true_false(o) for o in outs]