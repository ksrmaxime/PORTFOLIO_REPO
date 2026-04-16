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

--- STEP 1: READ THE LAW TITLE — CHOOSE YOUR EXAMINATION MODE ---

Look at the law title provided in the context and determine which examination mode applies.

MODE A — FULL EXAMINATION (all six targets)
Use this mode if the law primarily governs one of these development-side domains:
  - Data, personal information, or privacy
  - Computing, telecommunications, or digital infrastructure
  - Research, universities, or higher education
  - Financial markets, investment, or international trade
  - Intellectual property, copyright, or patents

In Mode A, examine the article against all six regulatory targets.

MODE B — USAGE-SIDE ONLY
Use this mode if the law governs any other domain (transport, health, environment, agriculture,
criminal law, social affairs, culture, civil procedure, etc.).

In Mode B, examine the article ONLY against INPUT and OUTPUT (targets 5 and 6).
A positive match on target 5 or 6 alone is sufficient to classify the article as TRUE.
Do not classify as TRUE for development-side targets (1–4) unless the article text itself
contains an explicit and direct reference to data collection infrastructure, computing systems,
technical education funding, technology investment, or hardware trade — not inferred from the sector.
Do not require a match on targets 1–4 before accepting a match on targets 5 or 6.

--- THE TWO-SIDED TEST ---

DEVELOPMENT SIDE (Mode A only) — ask: does this article DIRECTLY govern a resource that AI development depends on?
"Directly" means one single link: the article governs X, and X is a resource AI needs.
If the connection requires two or more steps (the article governs X → X relates to Y → Y is used by AI), it is too indirect → FALSE.

--- THE SIX REGULATORY TARGETS ---

DEVELOPMENT SIDE (examined in Mode A only):

1. SKILLS
   Does this article directly govern education, research, or public funding SPECIFICALLY in digital or technical fields?
   Qualifies for: digital skills programmes, computer science curricula, tech R&D funding, grants targeting innovation or technology sectors.
   Does not qualify for: general labour law or apprenticeship obligations applied to tech-adjacent companies, general education policy, professional licensing in non-technical fields.

2. COMPUTE RESOURCES
   Does this article directly govern the availability, conditions, or trade of COMPUTING hardware or infrastructure?
   Qualifies for: semiconductors and chips (production, trade, export controls), data centres, servers, cloud infrastructure, cybersecurity obligations specifically targeting computing infrastructure.
   Does not qualify for: general telecommunications networks or telephone services, general energy or construction rules, internet regulations with no specific link to computing infrastructure.

3. TRAINING DATA
   Does this article directly govern how data can be collected, accessed, or used at scale?
   Qualifies for: data protection rules on bulk or automated collection, data access rights, restrictions on mass scraping, rules on reuse of public or private datasets.
   Does not qualify for: copyright management oversight with no link to data collection, intellectual property enforcement rules whose object is counterfeit goods rather than data, general administrative data rules unrelated to large-scale processing.

4. CAPITAL
   Does this article directly govern financial flows or market conditions SPECIFICALLY in technology sectors?
   Qualifies for: tech-sector foreign investment screening, R&D tax incentives, technology-specific procurement rules, M&A rules in digital markets.
   Does not qualify for: general commercial, tax, or procurement law of broad application with no technology-specific dimension.

USAGE SIDE (examined in both modes, but with different criteria):

5. INPUT — what data flows into AI systems

   In MODE A (data, IP, telecom laws):
   Requires explicit reference to automated or AI processing.
   Qualifies for: automated data processing rules, algorithmic profiling, consent for AI-driven collection,
   biometric surveillance systems, cross-border transfers for automated systems.
   Does not qualify for: general privacy rules with no reference to automation.

   In MODE B (other sectoral laws):
   The article does NOT need to mention AI or automation explicitly.
   Qualifies if the article regulates a SENSITIVE DATA TYPE that is a primary AI input in this sector
   AND creates specific rights, restrictions, or obligations on how this data is collected or used.
   Sensitive data types that qualify: biometric data (facial images, fingerprints, voice), genetic or
   health data, behavioural data collected at scale (location, communications, browsing), financial
   transaction data at individual level.
   Does not qualify for: general professional confidentiality rules, administrative record-keeping with
   no individual-level or large-scale dimension, sector rules unrelated to data collection.

6. OUTPUT — what AI systems produce, decide, or do

   In MODE A (data, IP, telecom laws):
   Requires explicit reference to automated decisions or AI-generated content.
   Qualifies for: automated decision-making rules, AI-generated content liability, algorithmic accountability.
   Does not qualify for: general liability rules with no specific reference to automation.

   In MODE B (other sectoral laws):
   The article does NOT need to mention AI or automation explicitly.
   Qualifies if the article regulates either:
   (a) CONSEQUENTIAL DECISIONS affecting individuals in domains where AI systems produce such outcomes:
       hiring or dismissal, credit or insurance assessments, medical diagnosis or treatment decisions,
       administrative or judicial decisions, content moderation and publication;
   (b) the behaviour or authorisation of PHYSICAL AUTONOMOUS SYSTEMS acting without continuous
       human control: vehicles, robots, drones, or similar systems.
   AND the article creates specific obligations, rights, or liability rules around how these decisions
   or actions must be made, justified, challenged, or authorised.
   Does not qualify for: general safety standards applying equally to human and automated operators,
   sector rules that govern human professional conduct with no dimension of autonomous decision-making.

--- REASONING STEPS ---

STEP 1 — Choose examination mode
Read the law title. Does it indicate a development-side domain (data, computing, research, finance, IP, trade)? → Mode A (all six targets). Otherwise → Mode B (INPUT and OUTPUT only).

STEP 2 — Identify what the article governs
Describe its primary subject in plain terms, without projecting AI onto it.

STEP 3 — Apply the correct test
Mode A / development side: is there a direct, single-step link between what the article governs and a resource AI depends on?
Both modes / usage side: does the article specifically constrain the operation, inputs, or outputs of automated systems?

STEP 4 — Classify
The six targets are independent. ONE positive match on ANY applicable target is sufficient → TRUE.
For Mode B articles, finding a match only on targets 5 or 6 — with no match on targets 1–4 — is a fully valid TRUE outcome. Sectoral laws (transport, health, etc.) are expected to govern only the usage side.
CONTRADICTION GUARD: if your Step 3 analysis identified a positive match for any applicable target, you MUST output TRUE. Do not override a positive finding by noting that other targets were not matched.
If the link requires two or more steps, falls outside the applicable mode, or applies to a general category where AI is just one sub-case → FALSE.
If genuinely uncertain after Steps 1–3 → TRUE.

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
