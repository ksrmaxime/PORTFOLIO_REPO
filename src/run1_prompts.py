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

Determine whether this article governs at least one of the six targets below. Follow the three steps in order.

--- STEP 1: DELEGATION CHECK ---
Ask: does this article only state that an authority WILL regulate or decide something at a later stage?
If YES → FALSE, stop here. A future regulation may be relevant, but this article is not.
If the article itself grants a concrete authorisation ("authority X MAY do Y") or establishes a specific rule or obligation → it contains an instrument. Continue to Step 2.

--- STEP 2: SCOPE CHECK ---
Ask: does the law title indicate that this law primarily governs one of the five development-side nodes of the AI production chain?

The five development nodes are:
  - Supply of technical human capital and research (SKILLS)
  - Computing hardware and infrastructure (COMPUTE)
  - Data collection, access, and processing (TRAINING DATA)
  - Investment and financial flows in technology (CAPITAL)
  - Physical computing components and their trade (HARDWARE)

If YES → examine all six targets (1 to 6).
If NO → examine ONLY targets 5 (INPUT) and 6 (OUTPUT).

--- STEP 3: TARGET CHECK ---

1. SKILLS — Rules specifically funding or organising the supply of technical human capital or research.
   YES: digital skills programmes, tech R&D grants, computer science curricula, innovation subsidies.
   NO: general labour law, general apprenticeship obligations, non-technical professional licensing.

2. COMPUTE — Rules specifically governing computing hardware or infrastructure.
   YES: semiconductors, chips, export controls on computing components, data centres, servers, cloud, cybersecurity of computing systems.
   NO: general telecom or telephone services, general energy or construction rules.

3. TRAINING DATA — Rules governing large-scale data collection, access, or reuse.
   YES: data protection rules on collection and processing, data access rights, scraping restrictions, dataset reuse rules.
   NO: administrative record-keeping with no large-scale dimension, IP enforcement on counterfeit goods.

4. CAPITAL — Rules specifically governing financial flows in technology sectors.
   YES: tech-sector investment rules, foreign investment screening, R&D tax credits, tech-specific procurement.
   NO: general commercial law, general tax law, general procurement with no technology-specific dimension.

5. INPUT — Rules creating specific rights or restrictions on data that enters or feeds AI systems.
   YES if the article governs either:
   (a) data explicitly destined for automated or AI processing (automated profiling, AI-driven collection, algorithmic processing), OR
   (b) sensitive individual data — biometric, genetic, health, behavioural at scale, or financial — with specific rights or restrictions on how it is collected or used.
   NO: general professional confidentiality, administrative data rules with no individual-level or large-scale dimension.

6. OUTPUT — Rules creating specific obligations or liability around what automated or autonomous systems produce.
   YES if the article governs either:
   (a) decisions or content explicitly produced by automated or AI systems (algorithmic decisions, AI-generated content), OR
   (b) consequential decisions affecting individuals (hiring, credit, insurance, medical, administrative, judicial) OR physical autonomous systems (vehicles, robots, drones without continuous human control) — where the article establishes specific obligations, rights, or liability around how these decisions or actions are made.
   NO: general safety or liability rules applying equally to human and automated operators with no specific dimension of autonomous decision-making.

→ If at least one applicable target is governed → TRUE
→ If no applicable target is governed, or the link requires multiple inferential steps → FALSE
→ If genuinely uncertain → TRUE

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
