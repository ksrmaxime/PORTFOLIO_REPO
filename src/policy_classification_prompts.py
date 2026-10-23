# src/policy_classification_prompts.py
"""Step LLM 2 (v2, règles du 23.10.2026 : production = moyens de produire ; conditions = risque) : cible (cadran + sous-catégorie) et instrument des articles SPECIFIQUE.

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
    ("AI_PRODUCTION", "Encourager la production, sans conditions : l'État fournit ou facilite les moyens de produire l'IA (données d'entraînement, serveurs et puces pour l'entraînement, recherche, formation de développeurs)."),
    ("AI_ADOPTION", "Encourager l'adoption, sans conditions : l'État donne ou encourage directement l'utilisation (créer une cellule ou un réseau, coordonner, financer, aménager une infrastructure pour l'usage, permettre un usage sans le conditionner)."),
    ("AI_PRODUCTION_RISK", "Limiter les risques de la production : conditions ou exigences sur les moyens de produire l'IA (solidité du code : cybersécurité et gestion des mises à jour logicielles chez le développeur ; données d'entraînement ; infrastructure de calcul)."),
    ("AI_ADOPTION_RISK", "Limiter les risques de l'adoption : conditions, limites, exigences ou garanties sur le produit final, son utilisation, sa mise en circulation ou sa vente (usage autorisé sous conditions, homologation, exigences techniques sur le système mis en circulation, enregistrement des événements, contrôle humain, données traitées et leur effacement, sanctions, interdiction de vendre ou de fournir le service)."),
    ("", "Aucun mécanisme : l'article définit seulement un terme ou décrit l'objet de l'acte, sans règle propre."),
]

Q1_SYSTEM = """Tu es juriste, spécialiste du droit suisse et de la régulation de l'intelligence artificielle (IA).

L'article ci-dessous concerne un système d'IA ou un système qui en dépend (conduite automatisée, reconnaissance faciale…). Ta tâche : dire ce que l'État fait avec cet article, selon deux règles.

## Règle 1 — Risque ou non
Dès que l'article pose des conditions, des limites, des exigences ou des garanties, il limite les risques, MÊME s'il autorise l'utilisation. L'article n'est hors risque que si l'État donne ou encourage directement, sans conditions.

## Règle 2 — Production ou adoption
« Production » est réservé aux moyens de produire l'IA : données d'entraînement, solidité du code (cybersécurité, mises à jour logicielles chez le développeur), serveurs et puces pour l'entraînement, recherche. Tout ce qui porte sur le produit final relève de l'« adoption » : son utilisation, la sécurité de son utilisation, sa mise en circulation et son homologation, sa vente ou la fourniture de son service à des tiers, même quand l'obligation pèse sur le constructeur.

## Réponses possibles
{options}

## Remarques
- Une seule réponse, selon la fonction principale de l'article (premier alinéa et essentiel du texte).

## Exemples (inventés, pour illustrer le raisonnement)
- « La Confédération soutient des programmes de recherche en apprentissage automatique. » → A
- « L'office crée un centre de compétences pour accompagner les services qui utilisent l'IA. » → B
- « Le développeur doit protéger le code et les données d'entraînement de son modèle contre les intrusions. » → C
- « Le fabricant d'un robot de soins doit démontrer sa sécurité avant la mise sur le marché. » → D
- « L'office peut recourir à un logiciel d'analyse d'images si les personnes concernées en sont informées. » → D
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
        "SKILLS": "Compétences : cellule, réseau ou formation pour apprendre à utiliser l'IA, échanges d'expérience.",
        "RESEARCH_INNOVATION": "Recherche et innovation : projets d'application, développement de nouveaux usages de l'IA.",
        "INFRASTRUCTURE": "Infrastructures : plateformes, systèmes informatiques ou infrastructure physique qui permettent l'usage (signalisation, emplacements réservés).",
        "GOVERNANCE": "Gouvernance : habilitation d'une autorité à utiliser un système (sans conditions), coordination, organisation de la mise en œuvre.",
    },
    "AI_PRODUCTION_RISK": {
        "PRIVACY_RIGHTS": "Vie privée et droits : données personnelles utilisées pour l'entraînement, biais introduits à l'entraînement.",
        "SAFETY_SECURITY": "Sécurité : solidité du code (cybersécurité, gestion des mises à jour logicielles chez le développeur), sécurité de l'infrastructure d'entraînement.",
        "TRANSPARENCY_ACCOUNTABILITY": "Transparence et responsabilité : documentation du développement et des données d'entraînement.",
        "PROPERTY_ECONOMY": "Propriété et économie : droits d'auteur sur les données d'entraînement, concurrence entre fournisseurs.",
        "SOCIETY": "Société : effets collectifs de la production (environnement, consommation d'énergie du calcul).",
    },
    "AI_ADOPTION_RISK": {
        "PRIVACY_RIGHTS": "Vie privée et droits : données personnelles traitées par le système en usage, limitation des cas d'usage, consultation, effacement, profilage, non-discrimination.",
        "SAFETY_SECURITY": "Sécurité : exigences techniques sur le produit final, homologation, surveillance des systèmes en circulation, incidents, contrôles, entretien, conditions de sécurité d'une autorisation, restrictions d'usage, interdiction de vendre ou de fournir le service (sécurité nationale).",
        "TRANSPARENCY_ACCOUNTABILITY": "Transparence et responsabilité : enregistrement des événements (enregistreur), information des personnes et des utilisateurs, signalement de l'usage de l'IA, contrôle humain (surveillance par une personne, reprise du contrôle), attribution des responsabilités.",
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
- Le contrôle humain (surveillance par une personne, obligation de pouvoir reprendre le contrôle) et l'enregistrement des événements relèvent de la transparence et de la responsabilité, pas de la sécurité.

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
