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

USER_TEMPLATE = """You are classifying legal articles for a study on AI regulation.

== WHAT YOU ARE CLASSIFYING ==
You are evaluating a SINGLE ARTICLE, not the law as a whole.
Use the law title and chapter as context to understand the regulatory domain.
Base your decision on what THIS ARTICLE'S TEXT itself governs.
Different articles in the same law will have different answers.

== THE AI PRODUCTION CHAIN ==

To BUILD AI, four resources are needed:
- SKILLS: technical human capital — researchers, engineers, and the education and R&D funding that produces them.
- COMPUTE: physical infrastructure — chips, servers, data centres, cloud systems.
- TRAINING DATA: the data AI learns from — governed by rules on what data can be collected, accessed, or reused at scale.
- CAPITAL: financial flows — investment rules, R&D incentives, and tech-sector funding.

When AI is DEPLOYED, two interfaces are regulated:
- INPUT: data collected from individuals or fed into automated systems during use.
- OUTPUT: what AI systems produce — decisions affecting individuals, autonomous physical actions, AI-generated content, intellectual property of those outputs.

== CLASSIFICATION ==

Read the article text. Does it itself create a specific rule, right, or obligation — not merely delegate future regulation — that directly governs one of the following?

DEVELOPMENT SIDE — apply these only if the law title indicates the law governs that resource:

  SKILLS: rules specifically funding or organising technical education programmes, digital R&D, or innovation subsidies in technology sectors.

  COMPUTE: rules specifically governing computing hardware (chips, semiconductors), data centres, servers, cloud infrastructure, or cybersecurity of computing systems.

  TRAINING DATA: rules on data collection rights, bulk data access, processing authorisations, or reuse of datasets.

  CAPITAL: rules on investment, financing, or market conditions specifically targeting technology sectors.

USAGE SIDE — apply these regardless of the law's domain:

  INPUT: rules creating specific rights or restrictions on data collected from individuals or processed by automated systems — particularly biometric data (face, fingerprint, voice), genetic or health data, behavioural data collected at scale, or data explicitly destined for automated processing.

  OUTPUT: rules creating specific obligations, rights, or liability around what automated or autonomous systems produce — particularly decisions affecting individuals (hiring, credit, insurance, medical, administrative), autonomous physical systems (vehicles, robots, drones acting without continuous human control), AI-generated content, or intellectual property over outputs of automated systems.

An article that only states that an authority WILL regulate something later creates no rule of its own → FALSE.

→ TRUE if the article text directly governs at least one of the above elements.
→ FALSE if it governs something unrelated, or only delegates future regulation.

== LEGAL CONTEXT ==
Law: {law_title}
Chapter: {chapter_title}
Article: {article_title}

== ARTICLE TEXT ==
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
