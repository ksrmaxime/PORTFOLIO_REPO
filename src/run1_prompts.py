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

USER_TEMPLATE = """You are a legal expert classifying legal articles for a study on AI regulation.

== AI PRODUCTION CHAIN ==

Building AI requires four types of resources (DEVELOPMENT SIDE):
- SKILLS: the human capital that creates AI — researchers, engineers, and the education and funding that produces them.
- COMPUTE: the physical infrastructure AI runs on — chips, servers, data centres, and cloud systems.
- TRAINING DATA: the data AI learns from — governed by rules on what data can be collected, accessed, or reused at scale.
- CAPITAL: the financial flows that fund AI development — investment rules, R&D incentives, and technology-sector funding.

Deploying AI creates two regulatory interfaces (USAGE SIDE):
- INPUT: what goes into an AI system during use — data collected from individuals or sent to automated systems for processing.
- OUTPUT: what AI systems produce — decisions affecting individuals, autonomous physical actions, AI-generated content, or intellectual property of those outputs.

== CLASSIFICATION TASK ==

STEP 1 — Read the law title.
Does this law primarily govern one of the four development-side resources (skills, compute, training data, capital)?

If YES → go to STEP 2A.
If NO → go to STEP 2B.

---

STEP 2A — Development-side analysis.
Does this article itself create a specific rule, right, or obligation that directly governs one of the four resources?
Note: if the article only states that an authority will regulate the topic later, it creates no rule of its own → FALSE.

- SKILLS: specific rules on technical education programmes, digital R&D funding, or innovation subsidies targeted at technology sectors.
- COMPUTE: specific rules on computing hardware, data centres, servers, cloud infrastructure, or cybersecurity of computing systems.
- TRAINING DATA: rules on data collection rights, bulk data access, processing authorisations, or reuse of datasets.
- CAPITAL: rules on investment, funding, or market conditions specifically in technology sectors.

If the article governs one of these → TRUE.
If it only delegates future regulation, or governs something unrelated → FALSE.

---

STEP 2B — Usage-side analysis.
Does this article create a specific rule, right, or obligation that governs AI input or output?

- INPUT: rules on what data can be collected from individuals or processed by automated systems — in particular: biometric data (face, fingerprint, voice), genetic or health data, behavioural data collected at scale, or data explicitly processed by automated systems.
- OUTPUT: rules on what automated or autonomous systems produce or do — in particular: decisions affecting individuals (hiring, credit, medical, administrative), autonomous physical actions (vehicles, robots, drones acting without continuous human control), AI-generated content, or intellectual property rights over outputs produced by automated systems.

If the article governs one of these → TRUE.
If it governs something with no specific connection to AI input or output → FALSE.

== LEGAL CONTEXT ==
Law: {law_title}
Chapter: {chapter_title}
Article: {article_title}

== LEGAL TEXT ==
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
