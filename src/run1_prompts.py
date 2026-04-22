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

QUESTION: Does this article address at least one of the ten public problems below?

A provision qualifies if it represents a form of state intervention — binding or non-binding, financial, regulatory, or planning-oriented — that directly addresses one of these public problems. AI does not need to be mentioned explicitly.

A provision does NOT qualify if it only defines terms, states purposes, assigns regulatory competences to an authority without any substantive content of its own, or is a cross-reference that merely activates or applies other articles ("Article X applies", "the provisions of Y are applicable here").

The connection must be direct. "This article governs X, which AI may use or depend on" is not sufficient.

CLASSIFICATION RULE: if the article matches at least one public problem, answer TRUE. A match on a single problem is sufficient — do not require matches on multiple problems.

--- THE TEN PUBLIC PROBLEMS ---

DATA
1. Personal data protection — the state intervenes to govern how personal data is collected, processed, retained, shared, or transferred, because automated systems can exploit it in ways that harm citizens.
2. IP & creative content — the state intervenes to govern copyright or authorship rights, because generative systems can reproduce or exploit protected works without consent.

SKILLS
3. Education — the state intervenes to develop AI-relevant competencies through digital curricula, technical training programs, or digital literacy initiatives.
4. Research — the state intervenes to fund or organize scientific research in digital or AI-related fields, because markets may underprovide this capacity.

INFRASTRUCTURE
5. Compute & hardware — the state intervenes to govern the availability, procurement, or trade of chips, HPC systems, servers, or cloud infrastructure for AI development. Radio spectrum, satellite positions, and general telecommunications networks do not qualify.
6. Data centers & energy — the state intervenes to govern the construction, operation, security, or energy provisioning of data centers hosting AI infrastructure.

RISK & SOCIETAL HARMS
7. High-stakes application governance — the state intervenes to govern automated or autonomous systems, or automated decision-making processes, in sensitive domains.
   HARD RULE: if the article text does not contain an explicit reference to a system being automated, autonomous, or operating without human control — or to a decision being made algorithmically or by an automated process — target 7 cannot apply. No inference from the sector or domain is allowed.
   When this condition is met, qualifies if the article governs how such a system is authorised, tested, deployed, operated, supervised, or held liable.
   Does NOT qualify: general safety rules, infrastructure, advertising, speed limits, or any provision equally applicable to human operators — even if autonomous systems happen to operate in that sector.
8. Algorithmic accountability — the state intervenes to require explainability, auditability, or human oversight of automated decisions. Requires explicit reference to automated or algorithmic processing.
9. Disinformation — the state intervenes to govern synthetic media, deepfakes, or AI-generated misleading content. Requires explicit reference to generated or automated content.
10. Cybersecurity of AI systems — the state intervenes to protect the security or integrity of AI systems and their data pipelines specifically, distinct from general IT security.

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
