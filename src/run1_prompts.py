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

USER_TEMPLATE = """You are a legal expert building a cross-national comparative dataset of AI-relevant regulation.

Your task: determine whether this legal article governs at least one of the six regulatory targets below.

The correct test is DIFFERENT depending on which side of the AI production chain you are examining:

DEVELOPMENT SIDE — ask: does this article directly govern a resource that AI development depends on?
The article qualifies even if it was written before AI existed, and even if it applies to other sectors too.
What matters is whether the resource it governs (data, compute, skills, capital, hardware) is directly consumed by AI development.

USAGE SIDE — ask: does this article specifically constrain how an AI or automated system operates, receives inputs, or produces outputs?
General rules that apply equally to humans and machines do not qualify here.
What matters is whether the article creates obligations or rights specifically tied to automated processing or autonomous systems.

--- THE SIX REGULATORY TARGETS ---

DEVELOPMENT SIDE:

1. SKILLS
   Does this article govern education, research, training, or public funding in digital or technical fields?
   Qualifies even if it also covers non-AI sectors, as long as technical or digital skills and research are within its scope.

2. COMPUTE RESOURCES
   Does this article govern the availability, conditions, or trade of computing hardware or infrastructure?
   Qualifies for: semiconductors, chips, data centres, servers, cloud, digital networks, cybersecurity of computing infrastructure.
   Does not qualify for: general energy or construction rules with no specific link to computing.

3. TRAINING DATA
   Does this article govern how data can be collected, accessed, or used at scale?
   Qualifies for: data protection rules (they constrain what data can be used for training), bulk collection restrictions, data access rights, automated scraping rules.
   The article does not need to mention AI — governing data as a resource is sufficient.

4. CAPITAL
   Does this article govern financial flows or market conditions in technology sectors?
   Qualifies for: tech-sector investment rules, foreign investment screening, R&D tax incentives, technology procurement, M&A rules in digital markets.
   Does not qualify for: general commercial, tax, or procurement law with no technology-specific dimension.

USAGE SIDE:

5. INPUT
   Does this article specifically constrain what data can be collected or processed by automated or AI systems?
   Qualifies for: rules on automated data processing, algorithmic profiling, consent for AI-driven collection, biometric data for surveillance or recognition systems, cross-border transfer rules for data fed into AI.
   Does not qualify for: general privacy rules applying equally to manual human processing with no reference to automation.

6. OUTPUT
   Does this article specifically govern the results, decisions, or actions produced by automated or autonomous systems?
   Qualifies for: rules explicitly governing automated decisions, AI-generated content, autonomous vehicles or robots, liability for AI outputs, accountability requirements for algorithmic decision-making.
   Does not qualify for: general safety or liability rules applying to all operators (human or machine) that make no specific reference to automation.

--- REASONING STEPS ---

STEP 1 — Identify what the article governs
Describe its primary subject in plain terms.

STEP 2 — Apply the correct test
Development side: does this article govern a resource (data, compute, skills, capital, hardware) that AI directly depends on?
Usage side: does this article specifically constrain the operation, inputs, or outputs of automated systems?

STEP 3 — Classify
If yes to Step 2 for at least one target → TRUE
If no for all targets → FALSE
If genuinely uncertain → TRUE

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
