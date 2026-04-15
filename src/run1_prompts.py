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

--- STEP 1: PRELIMINARY FILTER — DELEGATION ARTICLES ---
If the article does nothing more than grant an authority the power to regulate a topic in the future
(e.g. "The Federal Council shall regulate...", "The authority may issue provisions on..."),
without itself creating any substantive rule, obligation, or right → classify as FALSE immediately.
The implementing regulation that follows may be relevant, but the delegation itself is not.

--- STEP 2: READ THE LAW TITLE — CHOOSE YOUR EXAMINATION MODE ---

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
Do not classify as TRUE for development-side targets (1–4) unless the article text itself
contains an explicit and direct reference to data collection infrastructure, computing systems,
technical education funding, technology investment, or hardware trade — not inferred from the sector.

--- THE TWO-SIDED TEST ---

DEVELOPMENT SIDE (Mode A only) — ask: does this article DIRECTLY govern a resource that AI development depends on?
"Directly" means one single link: the article governs X, and X is a resource AI needs.
If the connection requires two or more steps (the article governs X → X relates to Y → Y is used by AI), it is too indirect → FALSE.

USAGE SIDE (both modes) — ask: does this article SPECIFICALLY constrain how an AI or automated system operates, receives inputs, or produces outputs?
General rules that apply equally to humans and machines do not qualify.
What matters is whether the article creates obligations or rights specifically tied to automated processing or autonomous systems.

--- THE SIX REGULATORY TARGETS ---

DEVELOPMENT SIDE:

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

USAGE SIDE:

5. INPUT
   Does this article specifically constrain what data can be collected or processed by automated or AI systems?
   Qualifies for: rules on automated data processing, algorithmic profiling, consent requirements for AI-driven collection, biometric data for surveillance or recognition systems, cross-border transfer rules for data fed into AI.
   Does not qualify for: general privacy rules applying equally to manual human processing with no reference to automation.

6. OUTPUT
   Does this article specifically govern the results, decisions, or actions produced by automated or autonomous systems?
   Qualifies for: rules explicitly governing automated decisions, AI-generated content, autonomous vehicles or robots acting without human control, liability for AI outputs, accountability for algorithmic decision-making.
   Does not qualify for: general safety or liability rules applying to all operators (human or machine) with no specific reference to automation, sector regulations that predate automation and make no reference to it.

--- REASONING STEPS ---

STEP 1 — Delegation check
Does this article only grant power to regulate without itself creating a substantive rule? If yes → FALSE, stop here.

STEP 2 — Choose examination mode
Read the law title. Does it indicate a development-side domain (data, computing, research, finance, IP, trade)? → Mode A (all six targets). Otherwise → Mode B (INPUT and OUTPUT only).

STEP 3 — Identify what the article governs
Describe its primary subject in plain terms, without projecting AI onto it.

STEP 4 — Apply the correct test
Mode A / development side: is there a direct, single-step link between what the article governs and a resource AI depends on?
Both modes / usage side: does the article specifically constrain the operation, inputs, or outputs of automated systems?

STEP 5 — Classify
If yes to Step 4 for at least one applicable target → TRUE
If the link requires two or more steps, falls outside the applicable mode, or applies to a general category where AI is just one sub-case → FALSE
If genuinely uncertain after Steps 1–4 → TRUE

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
