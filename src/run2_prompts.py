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
USER_TEMPLATE = """Imagine you are a legal consultant advising companies that develop or use artificial intelligence systems today.
For each legal article, ask yourself one question:
Would this rule affect the decisions of an AI company today when developing, deploying, or operating AI systems?
Important reasoning rules:
Base your decision on the actual content of the rule, not on hypothetical future technologies.
TRUE only if the rule already applies in practice to activities typically performed by AI systems or to the data and digital processes they rely on.
FALSE if the rule concerns activities that would only become relevant to AI if a new or unusual system were invented.
If the connection to AI requires imagination or speculation, the answer must be FALSE.
Legal text:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)