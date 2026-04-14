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
USER_TEMPLATE = """You are a legal expert specializing in AI regulation.

Your task: determine whether a legal article directly regulates at least one of the following five variables. These variables are defined as the concrete regulatory targets that shape how AI is developed or used.

--- THE FIVE REGULATORY TARGETS ---

DEVELOPMENT SIDE:

1. SKILLS
   Rules governing education, academic research, or professional training in digital and technology fields.
   Rules on public funding, grants, or subsidies directed at technology development or innovation.
   Examples: university programs in computer science, research funding for tech sectors, digital skills curricula.

2. INFRASTRUCTURE
   Rules governing computing infrastructure relevant to AI: servers, data centres, cloud facilities, their security, maintenance, or national/federal deployment.
   Examples: regulations on data centre construction, national computing capacity, cybersecurity standards for critical digital infrastructure.

3. TRAINING DATA
   Rules governing the acquisition, access, or use of large-scale datasets for the purpose of training machine learning models.
   This covers data collection practices, licensing of datasets, or restrictions on mass data harvesting — NOT the individual inputs or outputs of a deployed AI system.
   Examples: rules on bulk web scraping, dataset licensing frameworks, obligations when collecting data for model training.

USAGE SIDE:

4. INPUT
   Rules that constrain or govern what information can be submitted to an AI system.
   This includes: personal data protection rules, cross-border data transfer restrictions, limits on what government agencies may process, and data collection by cameras or sensors feeding into integrated AI systems.
   Examples: privacy laws restricting what data users may enter, rules on biometric data capture for AI-powered surveillance.

5. OUTPUT
   Rules that govern what an AI system produces or the consequences of its outputs.
   This includes: AI-generated text, images, or creative works; decisions made by automated systems; actions taken by autonomous systems such as self-driving vehicles.
   Examples: liability rules for autonomous vehicle accidents, copyright rules for AI-generated content, accountability requirements for automated decisions.

--- CLASSIFICATION RULE ---
If the article directly regulates at least one of these five variables → TRUE
If the article does not regulate any of these variables → FALSE

Legal text:
{article_text}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(article_text=txt)