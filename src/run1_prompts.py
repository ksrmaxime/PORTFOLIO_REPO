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

Artificial intelligence means systems that use machine learning, neural networks, statistical models, or similar techniques to learn from data, make predictions, generate content, or automate decisions in ways that go beyond simple rule-based logic.

Important reasoning rules:
1. You must evaluate the object of the legal rule itself, not hypothetical future uses of AI.
2. CRITICAL: In Swiss legal texts, the abbreviation "AI" or "office AI" stands for "assurance-invalidité" (disability insurance) — this has NOTHING to do with artificial intelligence. Do NOT treat such references as AI-relevant.
3. General databases, registries, or administrative information systems are NOT AI systems, even if they process data electronically.

Classification rules:

TRUE only if the article specifically and directly governs one or more of the following:
- AI/ML systems themselves (algorithmic models, machine learning, neural networks, automated decision-making by algorithms)
- Autonomous or self-learning systems (e.g., autonomous vehicles, robotics with adaptive behavior)
- Algorithmic accountability, transparency, or explainability obligations
- Training data, datasets used for AI model development
- Automated profiling, scoring, or AI-generated recommendations with legal or significant effects

FALSE if the article governs any of the following, even if digital systems are involved:
- Physical infrastructure, traffic rules, construction, or public order
- Administrative registries, databases, or information systems used for record-keeping (not AI decision-making)
- Standard data protection rules that apply generally to any personal data, not specifically to AI
- Procedures, authorizations, or supervision of human experts (doctors, officers, etc.)
- Social insurance matters (disability insurance "AI/assurance-invalidité", old-age insurance "AVS", etc.)
- Copyright or intellectual property management not specifically tied to AI-generated content
- Any regulated activity where AI is only a hypothetical future application

Legal text:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)