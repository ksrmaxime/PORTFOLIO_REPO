# src/prompts.py (portfolio run2)
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = (
"You are a STRICT legal classification system.\n"
"You must respond with exactly TWO lines, in this order, without any additional text:\n"
"RELEVANT: TRUE or FALSE\n"
"JUSTIFICATION: a short justification (1–2 sentences) in English\n"
"Do not add anything else."
)
USER_TEMPLATE = """Imagine you are a legal consultant advising companies that develop or use artificial intelligence systems.
For each legal article, ask yourself the following question:
Would this rule potentially affect how a company designs, develops, deploys, secures, governs, or uses artificial intelligence systems?
Artificial intelligence should be understood broadly as systems that use algorithms and data to analyze information, make predictions, or automate tasks or decisions.
interpretation rule: A law is AI-relevant only if the regulated object itself could realistically involve algorithmic or automated data processing, not simply because an AI system could hypothetically interact with the regulated environment.

Important:
TRUE if the article establishes rules, obligations, responsibilities, procedures, restrictions, authorizations, or oversight that could affect AI systems or organizations developing or using them — whether the impact is direct or indirect.
Consider all domains where AI could be affected (data governance, digital infrastructure, automated decision-making, liability, security, compliance, etc.).
FALSE if the article clearly has no plausible impact on AI systems, their development, or their use.
Legal text:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)