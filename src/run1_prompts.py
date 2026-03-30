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
USER_TEMPLATE = """You are a legal expert specializing in AI law and policy.

Your task: determine whether a legal article has a DIRECT and SIGNIFICANT impact on the development, deployment, or governance of artificial intelligence — broadly understood.

AI should be understood in all its dimensions: technical systems, data ecosystems, digital infrastructure, economic conditions, societal frameworks, and fundamental rights that shape how AI is built and used.

--- STEP 1: Identify what the article actually regulates ---
Ask yourself: what is the core subject matter of this article? What activity, right, obligation, or relationship does it govern?

--- STEP 2: Apply the directness test ---
Ask yourself: would a lawyer specializing in AI law cite this article in a regulatory analysis of the AI sector? Would it appear in a compliance memo for a company developing or deploying AI?

An article has direct impact if it governs any of the following dimensions:

1. DATA — rules on collection, use, storage, sharing, or protection of personal or non-personal data that affect how AI systems are trained or operated
2. AUTOMATED SYSTEMS — rules on algorithmic decision-making, automated processing, autonomous systems, or AI-specific obligations (transparency, explainability, human oversight)
3. DIGITAL INFRASTRUCTURE — rules on computing infrastructure, cloud services, connectivity, or platforms that underpin AI systems
4. TECHNOLOGY POLICY — rules on public funding, research, digital transformation, or technology investment that shape the AI ecosystem
5. EDUCATION & WORKFORCE — rules on digital skills, AI literacy, research programs, or professional qualifications in relevant technical fields
6. INTELLECTUAL PROPERTY — rules on authorship, copyright, patents, or ownership that apply to AI-generated content, AI-assisted creation, or training data
7. FUNDAMENTAL RIGHTS & ETHICS — rules on non-discrimination, privacy, human dignity, or due process that directly constrain or shape how AI may be used
8. SECTOR-SPECIFIC AI APPLICATIONS — rules in sectors where AI plays a structurally important role (healthcare diagnostics, autonomous vehicles, financial scoring, content moderation, etc.)

An article does NOT have direct impact if:
- The connection to AI requires imagining a hypothetical future application not implied by the article itself
- It governs purely physical activities, human expert procedures, or administrative processes with no inherent computational dimension
- It is a generic administrative rule (inter-agency data sharing, organizational structure, procedural timelines) whose AI relevance is incidental
- Digital systems appear only as a delivery mechanism for a non-AI activity

--- STEP 3: Classify ---
If the article has a direct and significant impact on any dimension above → TRUE
If the article's connection to AI is absent, indirect, or merely hypothetical → FALSE

Legal text:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)