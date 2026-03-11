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
USER_TEMPLATE = """Decide whether this legal article regulates an activity that could realistically involve artificial intelligence systems.
Artificial intelligence should be understood broadly as systems that use algorithms and data to analyze information, make predictions, or automate tasks or decisions.
Important interpretation rule:
Do NOT imagine hypothetical AI applications. Only consider the activity that the law actually regulates.
Classification rule:
TRUE if the article regulates activities that inherently involve data processing, data analysis, information systems, automated decision-making, digital infrastructures, or analytical technologies where AI systems could realistically be used.
FALSE if the article regulates purely physical activities, infrastructure, public behavior, construction, traffic, administrative procedures, or other domains that do not inherently involve computational or data-processing systems.
Do not speculate about possible future AI uses. Base your decision only on the object of the legal rule.
Legal text:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)