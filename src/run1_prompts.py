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

Your task: determine whether this legal article directly governs at least one of the six regulatory targets below. These targets map onto the AI production chain — from the resources needed to build AI systems, to the conditions under which they operate.

--- CRITICAL ANCHORING RULE ---
An article qualifies ONLY IF its PRIMARY regulatory subject is one of the six targets.
Do NOT classify an article as TRUE simply because AI could theoretically be deployed in that sector.
The connection must exist in the article itself, not in a hypothetical future application.

Ask yourself: "Does this article govern the target directly, or does the link to AI only appear if I imagine an AI application that the article does not mention?"
If the link requires imagination → FALSE.

--- THE SIX REGULATORY TARGETS ---

DEVELOPMENT SIDE (resources needed to build AI):

1. SKILLS
   Rules whose primary subject is the supply of human capital for technology development.
   Covers: education and research programs in digital or technical fields; public funding for technology R&D; professional training in technical domains; grants or subsidies directed at technology or innovation sectors.
   Does NOT cover: general education policy, professional licensing in non-technical fields, vocational training unrelated to technology.

2. COMPUTE RESOURCES
   Rules whose primary subject is the availability or operating conditions of computing hardware or infrastructure.
   Covers: semiconductors and advanced chips — their production, trade, or export controls; data centres, servers, cloud infrastructure — their construction, operation, security, or energy requirements; network infrastructure underpinning digital services; cybersecurity obligations specifically targeting digital or computing infrastructure.
   Does NOT cover: general energy policy, general construction regulations, telecommunications rules that do not specifically concern computing infrastructure.

3. TRAINING DATA
   Rules whose primary subject is the acquisition of large-scale datasets for machine learning purposes.
   Covers: rights to collect, access, or use bulk data for model training; restrictions on mass data harvesting or automated scraping; dataset licensing and access frameworks.
   Does NOT cover: individual data protection rights in deployed systems (see INPUT), general statistical or administrative data collection unrelated to model training.

4. CAPITAL
   Rules whose primary subject is financial flows into AI and technology development.
   Covers: foreign investment screening or restrictions in technology sectors; rules on mergers and acquisitions involving technology firms; R&D tax incentives; public procurement rules that shape the technology market; regulations on technology company financing or market access.
   Does NOT cover: general corporate or commercial law, tax law of general application, standard public procurement rules not specific to technology.

USAGE SIDE (conditions under which AI systems operate):

5. INPUT
   Rules whose primary subject is what information can enter or be processed by a deployed AI system.
   Covers: personal data protection and consent requirements applicable to automated processing; cross-border data transfer restrictions; limits on what public bodies may process automatically; data collection by sensors, cameras, or connected devices feeding AI systems; rules on biometric or sensitive data used as system inputs.
   Does NOT cover: general privacy rules with no link to automated processing, surveillance rules applying exclusively to human operators.

6. OUTPUT
   Rules whose primary subject is the legal treatment of what AI or automated systems produce or decide.
   Covers: liability, copyright, or authorship rules for AI-generated content; accountability or oversight requirements for automated decisions affecting individuals; rules explicitly governing autonomous systems making real-world decisions (vehicles, robots, agents); sector-specific obligations explicitly triggered by AI-produced results.
   Does NOT cover: general traffic law governing human drivers (even in vehicles that could be automated), general liability rules of broad application with no AI-specific provision, sector regulations that predate AI and make no reference to automation.

--- REASONING STEPS ---

STEP 1 — Identify the primary subject
What is the core subject of this article? What activity, right, or obligation does it govern?
State it in plain terms without projecting AI onto it.

STEP 2 — Apply the anchoring test
Is this primary subject itself one of the six targets above?
Or does the connection to the targets only emerge by imagining an AI application the article does not mention?

STEP 3 — Classify
If the primary subject directly matches a target → TRUE
If the connection requires imagining an AI application → FALSE
If genuinely uncertain after Steps 1 and 2 → TRUE

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
