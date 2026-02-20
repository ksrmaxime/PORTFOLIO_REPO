from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


SYSTEM_PROMPT_FR = (
    "Tu es un système STRICT de classification juridique.\n"
    "Tu dois répondre UNIQUEMENT avec un JSON sur une seule ligne, sans texte autour.\n"
    "Le JSON doit contenir exactement ces deux clés:\n"
    '  - "relevant": true ou false\n'
    '  - "justification": une justification courte (1 à 3 phrases) en français\n'
    "Aucune autre clé. Aucune autre sortie."
)

USER_TEMPLATE_FR = """Tâche:
Décide si cet article de droit suisse fixe des règles, obligations, interdictions, procédures, responsabilités
ou exigences (techniques ou organisationnelles) pouvant influencer l’implémentation, le développement,
la sécurité, l’usage ou la responsabilité d’un système automatisé (au sens large), par exemple:
- logiciels / systèmes informatiques
- registres / bases de données
- infrastructures automatisées
- systèmes d’automatisation (y compris IA au sens large)

Important:
- Réponds TRUE (relevant=true) seulement si le texte impose/organise des effets normatifs concrets liés à ces systèmes
  (exigences, conditions, surveillance, responsabilités, procédures, obligations, autorisations, etc.).
- Réponds FALSE si le texte est une simple définition ou n’a pas de portée normative liée à ces systèmes.

Format de sortie obligatoire (JSON sur une ligne):
{{"relevant": true/false, "justification": "..." }}

Texte légal:
{article_text}
"""


def build_user_prompt(article_text: str) -> str:
    return USER_TEMPLATE_FR.format(article_text=(article_text or "").strip())


def _extract_json_object(s: str) -> Optional[str]:
    """
    Extract first JSON object {...} from a string (robust to stray tokens).
    We still require strict parse in the end.
    """
    if s is None:
        return None
    s = str(s).strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0).strip() if m else None


def parse_json_relevant_justif(raw: str) -> Tuple[Optional[bool], Optional[str], bool]:
    """
    Returns (relevant, justification, parse_ok)

    parse_ok=True only if:
    - valid JSON object
    - keys exactly {"relevant","justification"}
    - relevant is bool
    - justification is non-empty string
    """
    js = _extract_json_object(raw)
    if not js:
        return None, None, False

    try:
        obj = json.loads(js)
    except Exception:
        return None, None, False

    if not isinstance(obj, dict):
        return None, None, False

    keys = set(obj.keys())
    if keys != {"relevant", "justification"}:
        return None, None, False

    rel = obj.get("relevant", None)
    jus = obj.get("justification", None)

    if not isinstance(rel, bool):
        return None, None, False
    if not isinstance(jus, str) or not jus.strip():
        return None, None, False

    # Normalize whitespace in justification
    jus_clean = " ".join(jus.strip().split())
    return rel, jus_clean, True


@dataclass(frozen=True)
class ApertusConfig:
    model_path: str
    dtype: str = "bf16"          # "bf16" | "fp16"
    trust_remote_code: bool = True
    temperature: float = 0.0
    max_tokens: int = 160        # justification needs more room


class ApertusBatchClassifier:
    """
    Wrapper around TransformersClient.chat_many() with true batching.
    Returns:
      - relevant (Optional[bool])
      - justification (Optional[str])
      - raw output (str)
      - parse_ok (bool)
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
            rel, jus, ok = parse_json_relevant_justif(raw)
            relevants.append(rel)
            justifs.append(jus)
            oks.append(ok)

        return relevants, justifs, raws, oks