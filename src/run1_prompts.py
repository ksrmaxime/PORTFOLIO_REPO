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

USER_TEMPLATE = """You are a legal classifier. Follow the four steps below in order. Stop and output FALSE as soon as a step produces a negative result.

━━━ STEP 0 — INSTRUMENT FILTER ━━━
Does this article directly impose at least one of the following on any person, entity, or system?
  • An obligation (must, shall, is required to)
  • A prohibition (must not, is prohibited, may not)
  • A conditional right or permission (may, provided that…)
  • A liability rule or penalty

If NONE of the above → FALSE. Stop here.
The following never qualify, regardless of content:
  • Pure definitions ("for the purposes of this law, X means Y")
  • Scope or purpose clauses ("this law applies to…", "the aim of this law is…")
  • Bare competence assignments ("Authority X shall determine / regulate / set conditions for Y") that contain no substantive condition of their own

━━━ STEP 1 — EXAMINATION MODE ━━━
Does the law title unambiguously match one of these five domains — and no other?
  (A1) Data, personal information, or privacy
  (A2) Computing, telecommunications, or digital infrastructure
  (A3) Research, universities, or higher education
  (A4) Financial markets, investment, or international trade
  (A5) Intellectual property, copyright, or patents

If YES → Mode A (test all six targets in Step 3).
If NO or uncertain → Mode B (test only targets 5 and 6 in Step 3).
Transport, health, environment, agriculture, criminal law, labour, culture, civil procedure, construction, and all other sectoral domains → always Mode B.

━━━ STEP 2 — SUBJECT ━━━
State in one sentence what this article governs, without any reference to AI or automation.

━━━ STEP 3 — TARGET TEST ━━━

⚠ ANTI-INFERENCE RULE (applies in all modes):
A connection is valid only if the article directly governs the target.
"AI may use or depend on what this article governs" is NEVER sufficient — that reasoning is always indirect → do not use it.

— MODE A ONLY: DEVELOPMENT SIDE (targets 1–4) —
Apply each target as a one-step test: the article governs X, and X is the resource. Two or more steps → FALSE for that target.

  1. SKILLS — Does the article directly govern education, research, or public funding specifically in digital or technical fields?
     ✓ Digital skills programmes, computer science curricula, tech R&D funding, grants targeting innovation.
     ✗ General labour law, general education policy, professional licensing in non-technical fields.

  2. COMPUTE — Does the article directly govern the availability, conditions, or trade of computing hardware or infrastructure?
     ✓ Semiconductors, data centres, servers, cloud infrastructure, cybersecurity obligations for computing infrastructure.
     ✗ General telecom networks, general energy or construction rules, internet rules with no specific computing link.

  3. TRAINING DATA — Does the article directly govern how data is collected, accessed, or used at scale?
     ✓ Data protection for bulk or automated collection, data access rights, mass-scraping restrictions, dataset reuse rules.
     ✗ IP enforcement targeting counterfeit goods, general administrative data rules unrelated to large-scale processing.

  4. CAPITAL — Does the article directly govern financial flows or market conditions specifically in technology sectors?
     ✓ Tech-sector foreign investment screening, R&D tax incentives, technology-specific procurement, M&A rules in digital markets.
     ✗ General commercial, tax, or procurement law with no technology-specific dimension.

— BOTH MODES: USAGE SIDE (targets 5–6) —

  5. INPUT — Does the article impose specific obligations or restrictions on the collection or use of a sensitive data type?
     Sensitive data types: biometric data (facial images, fingerprints, voice), genetic or health data, large-scale behavioural data (location, communications, browsing), individual-level financial transaction data.
     Mode A: requires explicit reference to automated or AI processing.
     Mode B: does not require explicit AI reference, but the article must directly govern the sensitive data type — not merely mention it incidentally.
     ✗ General professional confidentiality, administrative record-keeping without individual-level or large-scale dimension.

  6. OUTPUT — Does the article specifically regulate what automated systems produce, decide, or do?
     Mode A: requires explicit reference to automated decisions or AI-generated content.
     Mode B: does not require explicit AI reference. Qualifies only if the article directly governs:
       (a) Consequential individual decisions in these domains: hiring or dismissal, credit or insurance assessments, medical diagnosis or treatment, administrative or judicial decisions, content moderation — AND the article imposes specific obligations on how such decisions are made, justified, or challenged; OR
       (b) The authorisation, operating conditions, or operator obligations of physical autonomous systems (vehicles, robots, drones) operating without continuous human control.
     ✗ General safety or technical standards that apply equally to human and automated operators.
     ✗ General infrastructure or environment rules that autonomous systems may incidentally use or depend on.

━━━ STEP 4 — CLASSIFY ━━━
One positive match on any applicable target → TRUE.
No positive match → FALSE.
CONTRADICTION GUARD: if Step 3 found a match on any target, output TRUE. Do not override a match by noting that other targets were not matched.
In Mode B, do not mention targets 1–4. Their absence is irrelevant and must not appear in your reasoning.

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
