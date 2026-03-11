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

For each legal article, you must ask yourself the following question:
Would this rule structurally affect how companies design, develop, deploy, operate, or govern AI systems?

Artificial intelligence should be understood broadly as systems that use algorithms and data to analyze information, make predictions, or automate tasks or decisions.

Important reasoning rule:
You must evaluate the object of the legal rule itself, not hypothetical future uses of AI.

Classification rules:

TRUE only if the article regulates activities that inherently involve data processing, data analysis, digital information systems, automated decision-making, computational infrastructure, or analytical technologies where AI systems could realistically operate.
TRUE if the rule establishes obligations, authorizations, supervision, procedures, or responsibilities directly linked to such digital or analytical activities.
FALSE if the article regulates primarily physical activities, infrastructure, traffic, construction, public behavior, or administrative procedures that do not inherently involve computational or data-processing systems.
Do NOT imagine hypothetical AI applications in unrelated domains.
In other words: classify TRUE only when the regulated activity itself is structurally connected to digital or analytical systems where AI could realistically be used.
Legal text:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)