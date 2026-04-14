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

Your task: determine whether this legal article directly governs at least one of the seven regulatory targets below. These targets map onto the AI production chain — from the resources needed to build AI systems, to the conditions under which they operate.

When in doubt, classify as TRUE. This is a first-pass filter: false positives are reviewed downstream, but false negatives are permanently lost.

--- THE SEVEN REGULATORY TARGETS ---

DEVELOPMENT SIDE (resources needed to build AI):

1. SKILLS
   Rules governing the supply of human capital for technology development.
   Covers: education and research programs in digital or technical fields; public funding for technology R&D; professional training and certification in technical domains; grants or subsidies directed at technology or innovation sectors.

2. INFRASTRUCTURE
   Rules governing the availability or operating conditions of computing and connectivity resources.
   Covers: data centres, servers, cloud infrastructure — their construction, operation, security, energy supply, or national deployment; network infrastructure underpinning digital services; cybersecurity obligations for critical digital infrastructure.

3. TRAINING DATA
   Rules governing the acquisition of large-scale datasets for machine learning purposes.
   Covers: rights to collect, access, or use bulk data for model training; restrictions on mass data harvesting or automated scraping; dataset licensing and access frameworks.
   Note: this covers data as a raw material for building models — not the individual information submitted by users to a deployed system (see INPUT).

4. CAPITAL
   Rules governing financial flows into AI and technology development.
   Covers: foreign investment screening or restrictions in technology sectors; rules on mergers and acquisitions involving technology firms; R&D tax incentives; public procurement rules that shape the technology market; regulations on technology company financing or market access.

5. HARDWARE
   Rules governing access to or trade in physical computing components.
   Covers: export controls or trade restrictions on semiconductors and advanced chips; regulations on procurement of hardware for AI or critical digital systems; restrictions on foreign supply of computing components.

USAGE SIDE (conditions under which AI systems operate):

6. INPUT
   Rules constraining what information can enter or be processed by a deployed AI system.
   Covers: personal data protection and consent requirements; cross-border data transfer restrictions; limits on what government or public agencies may process; data collection by sensors, cameras, or connected devices feeding AI systems; rules on biometric or sensitive data used as system inputs.

7. OUTPUT
   Rules governing the legal treatment of what AI systems produce or decide.
   Covers: liability, copyright, or authorship rules for AI-generated content (text, images, works); accountability or oversight requirements for automated decisions affecting individuals; rules on the actions of autonomous systems (vehicles, robots, agents); sector-specific obligations triggered by AI-produced results.

--- REASONING STEPS ---

STEP 1 — Functional mapping
What does this article actually control? Express it in functional terms: what resource, right, permission, or obligation does it create or restrict?

STEP 2 — Chain matching
Does the function identified in Step 1 directly affect any of the seven targets above?

STEP 3 — Classify
If yes to Step 2 → TRUE
If the connection requires a long causal chain, or the article governs something unrelated to these targets → FALSE
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
