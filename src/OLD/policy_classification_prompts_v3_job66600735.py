# src/policy_classification_prompts.py
"""Step LLM 2 (v3 : options présentées dans deux ordres et probabilités moyennées ; v2, règles du 23.10.2026 : production = moyens de produire ; conditions = risque) : cible (cadran + sous-catégorie) et instrument des articles SPECIFIQUE.

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
    ("AI_PRODUCTION", "Encourager la production, sans conditions : l'État fournit ou facilite les moyens de produire l'IA (données d'entraînement, serveurs et puces pour l'entraînement, recherche, appui au développement de l'IA comme un soutien en science des données et en IA, formation de développeurs)."),
    ("AI_ADOPTION", "Encourager l'adoption, sans conditions : l'État donne ou encourage directement l'utilisation (créer une cellule ou un réseau, coordonner, financer, aménager une infrastructure pour l'usage, permettre un usage sans le conditionner)."),
    ("AI_PRODUCTION_RISK", "Limiter les risques de la production : conditions ou exigences sur les moyens de produire l'IA eux-mêmes (données d'entraînement, infrastructure de calcul, code du modèle en cours de développement). Ne concerne jamais l'homologation, la réception par type ni les exigences sur un véhicule ou un système destiné à être utilisé."),
    ("AI_ADOPTION_RISK", "Limiter les risques de l'adoption : conditions, limites, exigences ou garanties sur le produit final, son utilisation, sa mise en circulation ou sa vente — y compris les obligations du constructeur sur ce produit (usage autorisé sous conditions, homologation et réception par type, exigences techniques sur le système, cybersécurité et mises à jour du produit, enregistrement des événements, notification des incidents, contrôle humain, données traitées et leur effacement, sanctions, interdiction de vendre ou de fournir le service)."),
    ("", "Aucun mécanisme : l'article définit seulement un terme ou décrit l'objet de l'acte, sans règle propre."),
]

Q1_SYSTEM = """Tu es juriste, spécialiste du droit suisse et de la régulation de l'intelligence artificielle (IA).

L'article ci-dessous concerne un système d'IA ou un système qui en dépend (conduite automatisée, reconnaissance faciale…). Ta tâche : dire ce que l'État fait avec cet article, selon deux règles.

## Règle 1 — Risque ou non
Dès que l'article pose des conditions, des limites, des exigences ou des garanties, il limite les risques, MÊME s'il autorise l'utilisation. L'article n'est hors risque que si l'État donne ou encourage directement, sans conditions.

## Règle 2 — Production ou adoption
« Production » est réservé aux moyens de produire l'IA eux-mêmes : données d'entraînement, code du modèle en cours de développement, serveurs et puces pour l'entraînement, recherche et appui au développement de l'IA. Tout ce qui porte sur le produit (un véhicule, un système, un service) relève de l'« adoption » : son utilisation, la sécurité de son fonctionnement, sa cybersécurité et ses mises à jour, son homologation et sa réception par type, sa mise en circulation, sa vente ou la fourniture de son service à des tiers — même quand l'obligation pèse sur le constructeur et même si elle s'applique avant la mise en circulation.

## Réponses possibles
{options}

## Remarques
- Une seule réponse, selon la fonction principale de l'article (premier alinéa et essentiel du texte).

## Exemples (inventés, pour illustrer le raisonnement)
{examples}

Réponds par une seule lettre : {letters}."""

Q1_EXAMPLES: List[Tuple[str, str]] = [
    ("La Confédération soutient des programmes de recherche en apprentissage automatique.", "AI_PRODUCTION"),
    ("L'office crée un centre de compétences pour accompagner les services qui utilisent l'IA.", "AI_ADOPTION"),
    ("Les données utilisées pour entraîner un modèle doivent être protégées contre les accès non autorisés.", "AI_PRODUCTION_RISK"),
    ("Le constructeur doit faire homologuer le système et assurer sa cybersécurité et ses mises à jour.", "AI_ADOPTION_RISK"),
    ("L'office peut recourir à un logiciel d'analyse d'images si les personnes concernées en sont informées.", "AI_ADOPTION_RISK"),
    ("Au sens de la présente loi, on entend par drone autonome un aéronef capable de voler sans pilote.", ""),
]

# ------------------------------------------------------------------ Q2 sous-cible
SUPPORT_SUBS = ["DATA", "SKILLS", "RESEARCH_INNOVATION", "INFRASTRUCTURE", "GOVERNANCE"]
RISK_SUBS = ["PRIVACY_RIGHTS", "SAFETY_SECURITY", "ACCOUNTABILITY_TRANSPARENCY", "PROPERTY_ECONOMY", "SOCIETY"]

SUB_DESC: Dict[str, Dict[str, str]] = {
    "AI_PRODUCTION": {
        "DATA": "Données : accès aux données d'entraînement (uniquement l'accès), partage de données pour développer des systèmes.",
        "SKILLS": "Compétences : formation de développeurs, de chercheurs et de spécialistes de l'IA.",
        "RESEARCH_INNOVATION": "Recherche et innovation : recherche en IA, instituts, programmes de développement, appui ou soutien au développement de l'IA (science des données, IA).",
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
        "SAFETY_SECURITY": "Sécurité : solidité du code pendant le développement, sécurité de l'infrastructure d'entraînement.",
        "ACCOUNTABILITY_TRANSPARENCY": "Responsabilité (accountability), puis transparence : pouvoir établir après coup comment le système a été développé et qui en répond — documentation du développement et des données d'entraînement.",
        "PROPERTY_ECONOMY": "Propriété et économie : droits d'auteur sur les données d'entraînement, concurrence entre fournisseurs.",
        "SOCIETY": "Société : effets collectifs de la production (environnement, consommation d'énergie du calcul).",
    },
    "AI_ADOPTION_RISK": {
        "PRIVACY_RIGHTS": "Vie privée et droits : données personnelles traitées par le système en usage, limitation des cas d'usage, consultation, effacement, profilage, non-discrimination.",
        "SAFETY_SECURITY": "Sécurité : exigences techniques sur le produit final, homologation, surveillance des systèmes en circulation, incidents, contrôles, entretien, conditions de sécurité d'une autorisation, restrictions d'usage, interdiction de vendre ou de fournir le service (sécurité nationale).",
        "ACCOUNTABILITY_TRANSPARENCY": "Responsabilité (accountability), puis transparence. Responsabilité : pouvoir établir après coup qui a fait quoi et qui en répond — enregistrement des événements (enregistreur), traçabilité, accès aux données enregistrées pour établir les faits après un accident, contrôle humain (surveillance par une personne, reprise du contrôle), désignation des responsables, instructions et explications remises aux utilisateurs. Transparence, en second lieu : information des personnes, signalement de l'usage de l'IA.",
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
- Responsabilité (accountability) : dès que l'article sert à savoir après coup qui a fait quoi et qui en répond (enregistrement des événements, traçabilité, contrôle humain, désignation d'un responsable, instructions remises aux utilisateurs), la réponse est « Responsabilité (accountability), puis transparence », pas « Sécurité », même si le but final est la sécurité.

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
{examples}

Réponds par une seule lettre : {letters}."""

Q3_EXAMPLES: List[Tuple[str, str]] = [
    ("L'agence nationale est chargée de surveiller les systèmes d'IA utilisés par l'administration.", "ORGANISATION"),
    ("La police peut utiliser des caméras à reconnaissance faciale lors de grands événements.", "AUTHORIZATION"),
    ("L'exploitant doit informer les clients qu'ils dialoguent avec un agent conversationnel.", "OBLIGATION"),
    ("La vente de jouets connectés capables d'enregistrer les conversations est interdite.", "PROHIBITION_BAN"),
    ("Est puni d'une amende quiconque met en service un drone autonome sans homologation.", "LIABILITY"),
    ("Les cantons reçoivent une contribution aux frais d'acquisition du système.", "TAXES_SUBSIDIES"),
    ("L'office mène des essais pilotes et publie un rapport d'évaluation.", "PLANNING_EVALUATION"),
]

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


def _render(items: List[Tuple[str, str]], order: List[int]):
    """items = [(code, description)] ; order = permutation des indices. Retourne le bloc
    d'options, les lettres et les codes dans l'ordre présenté."""
    letters = LETTERS[: len(items)]
    codes = [items[k][0] for k in order]
    block = "\n".join(f"{l} — {items[k][1]}" for l, k in zip(letters, order))
    return block, letters, codes


def _examples(examples: List[Tuple[str, str]], letters: List[str], codes: List[str]) -> str:
    pos = {c: l for l, c in zip(letters, codes)}
    return "\n".join(f"- « {t} » → {pos[c]}" for t, c in examples if c in pos)


def orders(n: int, debias: bool) -> List[List[int]]:
    """Ordres de présentation des options : l'ordre de référence et, si debias, un décalage
    circulaire d'une demi-liste (chaque option change de position et de lettre, y compris
    celle du milieu, ce qu'un simple renversement ne ferait pas pour un nombre impair)."""
    base = list(range(n))
    k = (n + 1) // 2
    return [base, base[k:] + base[:k]] if debias else [base]


def q1_system(order: List[int] = None):
    order = order or list(range(len(QUADRANTS)))
    block, letters, codes = _render(QUADRANTS, order)
    return Q1_SYSTEM.format(options=block, letters=", ".join(letters),
                            examples=_examples(Q1_EXAMPLES, letters, codes)), letters, codes


def q2_system(quadrant: str, order: List[int] = None):
    subs = RISK_SUBS if quadrant.endswith("_RISK") else SUPPORT_SUBS
    items = [(c, SUB_DESC[quadrant][c]) for c in subs]
    order = order or list(range(len(items)))
    block, letters, codes = _render(items, order)
    qdesc = dict(QUADRANTS)[quadrant]
    return Q2_SYSTEM.format(quadrant_desc=qdesc, focus=FOCUS[quadrant], options=block,
                            letters=", ".join(letters)), letters, codes


def q3_system(order: List[int] = None):
    order = order or list(range(len(INSTRUMENTS)))
    block, letters, codes = _render(INSTRUMENTS, order)
    return Q3_SYSTEM.format(options=block, letters=", ".join(letters),
                            examples=_examples(Q3_EXAMPLES, letters, codes)), letters, codes


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
