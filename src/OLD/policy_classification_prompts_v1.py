# src/policy_classification_prompts.py
"""Step LLM 2 : cible (cadran + sous-catégorie) et instrument des articles SPECIFIQUE.

Cf. le codebook (Claude Docs « Codebook – Portfolio IA »). Trois questions, chacune posée avec
toutes ses options à la fois, réponse en une lettre (lecture de la probabilité des lettres) :
  Q1 cadran      : A=AI_PRODUCTION, B=AI_ADOPTION, C=AI_PRODUCTION_RISK, D=AI_ADOPTION_RISK,
                   E=aucun mécanisme (définition / objet)
  Q2 sous-cible  : 5 options, qui dépendent de la colonne du cadran (encourager / limiter) et
                   dont la description dépend de la ligne (amont / aval)
  Q3 instrument  : 9 options, indépendamment de la cible
Les exemples sont inventés (aucun article du corpus) : ils illustrent le raisonnement.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

LETTERS = list("ABCDEFGHI")

# ------------------------------------------------------------------ Q1 cadran
QUADRANTS: List[Tuple[str, str]] = [
    ("AI_PRODUCTION", "Encourager en amont (production) : recherche en IA, accès aux données d'entraînement, capacités de calcul, formation de développeurs et de chercheurs, cadre favorable au développement."),
    ("AI_ADOPTION", "Encourager en aval (adoption) : permettre, autoriser ou organiser l'utilisation d'un système (habilitation d'une autorité, procédure d'autorisation d'usage, coordination, projets pilotes et essais, infrastructures pour l'usage, formation des utilisateurs)."),
    ("AI_PRODUCTION_RISK", "Limiter les risques en amont : exigences sur le système en tant que produit et obligations de ceux qui le conçoivent, le construisent ou le fournissent (exigences techniques, tests, homologation, documentation, enregistrement des événements, cybersécurité, notification des incidents par le constructeur, restrictions à la fourniture de systèmes ou de modèles)."),
    ("AI_ADOPTION_RISK", "Limiter les risques en aval : conditions et limites de l'utilisation et obligations de ceux qui utilisent ou exploitent le système (cas d'usage autorisés, données traitées et leur effacement, information des personnes, contrôle humain, sécurité d'exploitation, sanctions des utilisateurs)."),
    ("", "Aucun mécanisme : l'article définit seulement un terme ou décrit l'objet de l'acte, sans règle propre."),
]

Q1_SYSTEM = """Tu es juriste, spécialiste du droit suisse et de la régulation de l'intelligence artificielle (IA).

L'article ci-dessous concerne un système d'IA ou un système qui en dépend (conduite automatisée, reconnaissance faciale…). Ta tâche : dire ce que l'État fait avec cet article, selon deux axes.

## Axe 1 — Encourager ou limiter les risques
- Encourager : l'article ouvre, permet, facilite, finance ou organise le développement ou l'utilisation de l'IA.
- Limiter les risques : l'article pose des conditions, des limites, des exigences ou des garanties (sécurité, protection des données, transparence, contrôle, interdiction, sanction).

## Axe 2 — Amont ou aval
- Amont (production) : l'article vise le système en tant que produit et ceux qui le développent, le construisent, l'entraînent, l'homologuent ou le fournissent.
- Aval (utilisation) : l'article vise l'utilisation du système et ceux qui l'utilisent ou l'exploitent (autorités, conducteurs, opérateurs, exploitants, détenteurs).

## Réponses possibles
{options}

## Règles
- Une seule réponse, selon la fonction principale de l'article (premier alinéa et essentiel du texte).
- Une habilitation qui permet à une autorité d'utiliser un système relève de B, même si l'article fixe aussi quelques limites. Un article consacré aux limites de cet usage (cas d'usage, données, consultation, effacement) relève de D.
- Une procédure de demande d'autorisation relève de B ; les conditions de sécurité à remplir pour l'obtenir relèvent de D, ou de C si elles portent sur le produit lui-même.

## Exemples (inventés, pour illustrer le raisonnement)
- « La Confédération soutient des programmes de recherche en apprentissage automatique. » → A
- « L'office peut recourir à un logiciel d'analyse d'images pour trier les dossiers. » → B
- « Le fabricant d'un robot de soins doit démontrer sa sécurité avant la mise sur le marché. » → C
- « Les images saisies par le système de reconnaissance sont effacées après dix jours. » → D
- « Au sens de la présente loi, on entend par drone autonome un aéronef capable de voler sans pilote. » → E

Réponds par une seule lettre : {letters}."""

# ------------------------------------------------------------------ Q2 sous-cible
SUPPORT_SUBS = ["DATA", "SKILLS", "RESEARCH_INNOVATION", "INFRASTRUCTURE", "GOVERNANCE"]
RISK_SUBS = ["PRIVACY_RIGHTS", "SAFETY_SECURITY", "TRANSPARENCY_ACCOUNTABILITY", "PROPERTY_ECONOMY", "SOCIETY"]

SUB_DESC: Dict[str, Dict[str, str]] = {
    "AI_PRODUCTION": {
        "DATA": "Données : accès aux données d'entraînement, partage de données pour développer des systèmes.",
        "SKILLS": "Compétences : formation de développeurs, de chercheurs et de spécialistes de l'IA.",
        "RESEARCH_INNOVATION": "Recherche et innovation : recherche en IA, instituts, programmes de développement.",
        "INFRASTRUCTURE": "Infrastructures : capacités de calcul, supercalculateurs, plateformes de développement.",
        "GOVERNANCE": "Gouvernance : sécurité juridique, coordination ou cadre favorable au développement.",
    },
    "AI_ADOPTION": {
        "DATA": "Données : données et interopérabilité nécessaires à l'utilisation des systèmes.",
        "SKILLS": "Compétences : formation des utilisateurs, opérateurs ou du personnel qui emploie les systèmes, échanges d'expérience.",
        "RESEARCH_INNOVATION": "Recherche et innovation : projets pilotes, essais, développement de nouvelles applications.",
        "INFRASTRUCTURE": "Infrastructures : plateformes, systèmes informatiques ou infrastructure physique qui permettent l'usage (signalisation, emplacements réservés).",
        "GOVERNANCE": "Gouvernance : habilitation d'une autorité à utiliser un système, procédure d'autorisation d'usage, coordination, organisation et déroulement de l'usage.",
    },
    "AI_PRODUCTION_RISK": {
        "PRIVACY_RIGHTS": "Vie privée et droits : données personnelles utilisées pour l'entraînement, biais introduits à la conception.",
        "SAFETY_SECURITY": "Sécurité : exigences techniques, tests, homologation, certification, cybersécurité, surveillance des incidents, restrictions à la fourniture de systèmes ou de modèles (y compris pour des raisons de sécurité nationale).",
        "TRANSPARENCY_ACCOUNTABILITY": "Transparence et responsabilité : documentation du système, enregistrement des événements, indication des conditions d'utilisation prévues par le constructeur.",
        "PROPERTY_ECONOMY": "Propriété et économie : droits d'auteur sur les données d'entraînement, concurrence entre fournisseurs.",
        "SOCIETY": "Société : effets collectifs de la production (environnement, consommation d'énergie du calcul).",
    },
    "AI_ADOPTION_RISK": {
        "PRIVACY_RIGHTS": "Vie privée et droits : données traitées par le système, limitation des cas d'usage, consultation, effacement, profilage, non-discrimination.",
        "SAFETY_SECURITY": "Sécurité : sécurité d'exploitation, contrôles, entretien, restrictions d'usage, sanctions liées à la sécurité.",
        "TRANSPARENCY_ACCOUNTABILITY": "Transparence et responsabilité : information des personnes, signalement de l'usage de l'IA, contrôle humain (surveillance par une personne, reprise du contrôle), attribution des responsabilités.",
        "PROPERTY_ECONOMY": "Propriété et économie : droits sur les contenus générés, effets économiques de l'usage.",
        "SOCIETY": "Société : hypertrucages, désinformation, effets sur le travail ou la démocratie.",
    },
}

Q2_SYSTEM = """Tu es juriste, spécialiste du droit suisse et de la régulation de l'intelligence artificielle (IA).

L'article ci-dessous a été classé dans la catégorie suivante : {quadrant_desc}

Ta tâche : préciser {focus}.

## Réponses possibles
{options}

## Règles
- Une seule réponse, selon la fonction principale de l'article.
- Le contrôle humain (surveillance par une personne, obligation de pouvoir reprendre le contrôle) relève de la transparence et de la responsabilité, pas de la sécurité.

Réponds par une seule lettre : {letters}."""

FOCUS = {"AI_PRODUCTION": "la ressource que l'État fournit ou facilite par cet article",
         "AI_ADOPTION": "la ressource que l'État fournit ou facilite par cet article",
         "AI_PRODUCTION_RISK": "l'intérêt que cet article protège",
         "AI_ADOPTION_RISK": "l'intérêt que cet article protège"}

# ------------------------------------------------------------------ Q3 instrument
INSTRUMENTS: List[Tuple[str, str]] = [
    ("ORGANISATION", "Organisation : confier une tâche ou un rôle à une autorité ou un organisme, ou en créer un (coordination, surveillance, réseau, groupe de suivi)."),
    ("AUTHORIZATION", "Autorisation : base légale qui PERMET — habilitation d'une autorité (« peut »), régime d'autorisation, d'admission ou d'homologation, usage permis sous conditions."),
    ("OBLIGATION", "Obligation : exigence imposée à un acteur (« doit », « est tenu de »), y compris le devoir qui correspond à un droit des personnes."),
    ("PROHIBITION_BAN", "Interdiction : interdire une activité, restreindre un usage hors d'un cadre autorisé, clauses de sauvegarde."),
    ("LIABILITY", "Responsabilité et sanctions : désigner qui répond de quoi, amendes, peines."),
    ("TAXES_SUBSIDIES", "Taxes et subventions : contributions financières, subventions, taxes."),
    ("PUBLIC_INVESTMENT", "Investissement public : l'État finance ou construit lui-même (infrastructure, programme)."),
    ("PLANNING_EVALUATION", "Planification et évaluation : stratégie, rapports, évaluation, essais pilotes et expérimentations."),
    ("VOLUNTARY", "Instruments non contraignants : recommandations, information, labels, coopération facultative (« peuvent collaborer »)."),
]

Q3_SYSTEM = """Tu es juriste, spécialiste du droit suisse et de la régulation de l'intelligence artificielle (IA).

L'article ci-dessous concerne un système d'IA ou un système qui en dépend. Ta tâche : identifier l'INSTRUMENT, c'est-à-dire le mécanisme juridique par lequel l'article agit (comment), quel que soit son objet.

## Réponses possibles
{options}

## Règles
- Une seule réponse : le mécanisme principal de l'article.
- « peut », « est autorisé », « ne peut être admis que si » → autorisation ; « doit », « est tenu de » → obligation.
- Un article qui fixe la procédure ou les conditions d'une autorisation relève de l'autorisation.

## Exemples (inventés, pour illustrer le raisonnement)
- « L'agence nationale est chargée de surveiller les systèmes d'IA utilisés par l'administration. » → A
- « La police peut utiliser des caméras à reconnaissance faciale lors de grands événements. » → B
- « L'exploitant doit informer les clients qu'ils dialoguent avec un agent conversationnel. » → C
- « La vente de jouets connectés capables d'enregistrer les conversations est interdite. » → D
- « Est puni d'une amende quiconque met en service un drone autonome sans homologation. » → E
- « Les cantons reçoivent une contribution aux frais d'acquisition du système. » → F
- « L'office mène des essais pilotes et publie un rapport d'évaluation. » → H

Réponds par une seule lettre : {letters}."""

# ------------------------------------------------------------------ prompt utilisateur
USER_TEMPLATE = """Acte : {act_title}{act_meta}
Emplacement : {breadcrumb}
{definitions}Article : {article_label}{marginal}

Texte de l'article :
{article_text}

Système concerné (terme repéré) : {terms}

{question}"""

Q_LINES = {
    "q1": "Que fait l'État avec cet article ? Réponds par une seule lettre : {letters}.",
    "q2": "Quelle sous-catégorie ? Réponds par une seule lettre : {letters}.",
    "q3": "Quel est l'instrument ? Réponds par une seule lettre : {letters}.",
}

JUSTIF_SUFFIX = "\n\nJustification en une phrase, en tes propres mots, en citant le passage décisif de l'article : "


def _options(items: List[str]) -> Tuple[str, List[str]]:
    letters = LETTERS[: len(items)]
    return "\n".join(f"{l} — {t}" for l, t in zip(letters, items)), letters


def q1_system() -> Tuple[str, List[str]]:
    opts, letters = _options([d for _, d in QUADRANTS])
    return Q1_SYSTEM.format(options=opts, letters=", ".join(letters)), letters


def q2_system(quadrant: str) -> Tuple[str, List[str], List[str]]:
    subs = RISK_SUBS if quadrant.endswith("_RISK") else SUPPORT_SUBS
    opts, letters = _options([SUB_DESC[quadrant][s] for s in subs])
    qdesc = dict(QUADRANTS)[quadrant]
    return Q2_SYSTEM.format(quadrant_desc=qdesc, focus=FOCUS[quadrant], options=opts, letters=", ".join(letters)), letters, subs


def q3_system() -> Tuple[str, List[str]]:
    opts, letters = _options([d for _, d in INSTRUMENTS])
    return Q3_SYSTEM.format(options=opts, letters=", ".join(letters)), letters


def build_user_prompt(row, ctx: Dict[str, str], terms: List[str], definitions: str, question: str, letters: List[str]) -> str:
    meta = []
    if str(row.get("sr_number", "") or "").strip():
        meta.append(f"RS {row['sr_number']}")
    if str(row.get("doc_type_label", "") or "").strip():
        meta.append(str(row["doc_type_label"]))
    return USER_TEMPLATE.format(
        act_title=ctx["act_title"] or "(titre inconnu)",
        act_meta=f" ({', '.join(meta)})" if meta else "",
        breadcrumb=ctx["breadcrumb"],
        definitions=("\n" + definitions) if definitions else "",
        article_label=str(row.get("label", "")).strip(),
        marginal=f" — {ctx['marginal']}" if ctx["marginal"] else "",
        article_text=str(row.get("text", "") or "").strip(),
        terms=", ".join(f"« {t} »" for t in terms) if terms else "(non précisé)",
        question=Q_LINES[question].format(letters=", ".join(letters)),
    )
