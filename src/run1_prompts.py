# src/prompts.py (portfolio run1)
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = (
"You are a STRICT legal classification system.\n"
"You must respond with exactly TWO lines, in this order, without any additional text:\n"
"RELEVANT: TRUE or FALSE\n"
"JUSTIFICATION: a short justification (1–2 sentences) in English\n"
"Do not add anything else."
)

USER_TEMPLATE = """You are classifying legal articles for an AI regulation portfolio.

QUESTION: Does this article substantively address at least one of the ten public problems below?

Answer FALSE immediately if the article only: defines terms, states a purpose or scope, delegates regulatory power without imposing any substantive condition, or cross-references other articles without adding content of its own.
One match on any problem is sufficient for TRUE.

--- THE TEN PUBLIC PROBLEMS ---

DATA
1. Personal data protection — governing how personal data is collected, processed, retained, shared, or transferred.
2. IP & creative content — governing copyright or authorship rights for content that may be generated or processed by automated systems.

SKILLS
3. Education — developing AI-relevant competencies through digital curricula, technical training, or digital literacy. Must be specifically in digital or technical fields — general education does not qualify.
4. Research — funding or organizing scientific research specifically in digital or AI-related fields.

INFRASTRUCTURE
5. Compute & hardware — governing the availability, procurement, or trade of chips, HPC systems, servers, or cloud infrastructure. Radio spectrum, satellite positions, and general telecommunications networks do NOT qualify.
6. Data centers & energy — governing the construction, operation, security, or energy provisioning of data centers.

RISK & SOCIETAL HARMS
7. High-stakes application governance — governing AI outputs: automated or autonomous systems (e.g. vehicles without a driver, systems with automation), or automated decision-making processes.
   The article text must explicitly name or describe one of these AI outputs. If the article text contains no such explicit reference, target 7 does not apply — even if the sector (transport, health, etc.) is one where AI is used.
   When the explicit reference is present: qualifies if the article imposes any condition, restriction, obligation, or liability specifically on how that system or decision is authorised, tested, deployed, operated, or supervised.
8. Algorithmic accountability — requiring explainability, auditability, or human oversight of automated decisions. Must explicitly reference automated or algorithmic processing.
9. Disinformation — governing synthetic media, deepfakes, or AI-generated misleading content. Must explicitly reference generated or automated content.
10. Cybersecurity of AI systems — governing the security or integrity of AI systems and their data pipelines, distinct from general IT security.

--- LEGAL CONTEXT ---
Law: {law_title}
Chapter: {chapter_title}
Article: {article_title}

--- LEGAL TEXT ---
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()

    def _get(col: str) -> str:
        val = row.get(col, "")
        return "" if pd.isna(val) else str(val).strip()

    return USER_TEMPLATE.format(
        law_title=_get("ctx_law_title") or "N/A",
        chapter_title=_get("ctx_chapter_title") or "N/A",
        article_title=_get("ctx_article_title") or "N/A",
        article_text=txt,
    )
