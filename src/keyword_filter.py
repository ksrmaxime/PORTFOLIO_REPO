"""Filtrage par mots-clés du dataset Fedlex (sortie de download_1.py).

Principe
--------
1. Chaque ligne de niveau 1 à 6 (titres de parties / chapitres / sections et articles)
   est normalisée (minuscules, sans accents, apostrophes et espaces unifiés), puis
   comparée à une liste de motifs regex groupés par thème (KEYWORDS).
2. Un acte est retenu si AU MOINS UNE de ses lignes contient un motif.
3. La sortie garde TOUTES les lignes des actes retenus (structure complète, pour garder
   le contexte), avec des colonnes qui indiquent quelles lignes ont matché et quoi.

Les motifs sont volontairement séparés en deux niveaux de précision :
  - "core"     : vocabulaire explicite de l'IA (peu de faux positifs) ;
  - "extended" : automatisation / décision automatisée / biométrie / algorithmes, plus
                 large, utile pour ne rien rater mais à trier ensuite.
Chaque ligne et chaque acte portent les groupes et le niveau qui ont matché, ce qui
permet de filtrer plus finement en aval sans relancer la recherche.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


# ============================================================
# Mots-clés
# ============================================================
# Les motifs s'appliquent au texte NORMALISÉ (minuscules, sans accents, apostrophe ' simple,
# espaces simples). Écrire les motifs sans accents. `case_sensitive=True` : le motif est
# appliqué au texte sans accents mais AVEC la casse d'origine (pour les sigles comme "IA").

@dataclass(frozen=True)
class Keyword:
    group: str          # thème
    tier: str           # "core" | "extended"
    label: str          # forme lisible, reportée dans les colonnes de sortie
    pattern: str        # regex sur texte normalisé
    case_sensitive: bool = False


_S = r"s?"          # pluriel optionnel
_E = r"e?s?"        # accord optionnel (automatise / automatisee / automatises / automatisees)

KEYWORDS: List[Keyword] = [
    # --- IA explicite -------------------------------------------------------
    Keyword("ia", "core", "intelligence artificielle", rf"\bintelligences? artificielles?\b"),
    # sigle seulement précédé d'un article élidé ("l'IA", "d'IA") : "IA" seul attrape
    # "annexes IA et IB" (règlement CE) ou "IA-System" (Imagery Analyst)
    Keyword("ia", "core", "IA (sigle : l'IA, d'IA)", r"(?:\bl'|\bd')IA\b(?![-\w])", case_sensitive=True),
    Keyword("ia", "core", "système d'IA", rf"\bsystemes? d'(?:ia|intelligence artificielle)\b"),
    Keyword("ia", "core", "IA générative / modèle de langage",
            rf"\b(?:ia generative|intelligence artificielle generative|grands? modeles? de langage|modeles? de langage|modeles? d'ia)\b"),
    # --- Apprentissage automatique -----------------------------------------
    Keyword("apprentissage", "core", "apprentissage automatique / machine learning",
            r"\b(?:apprentissage automatique|apprentissage machine|machine learning)\b"),
    Keyword("apprentissage", "core", "apprentissage profond / réseau de neurones",
            rf"\b(?:apprentissage profond|deep learning|reseaux? (?:de )?neurone{_S}|reseaux? neuron(?:al|aux))\b"),
    # --- Contenus synthétiques ---------------------------------------------
    Keyword("contenus_synthetiques", "core", "hypertrucage / deepfake",
            r"\b(?:hypertrucages?|deep ?fakes?|contenus? (?:genere|synthetique)s? (?:par|au moyen))"),
    Keyword("contenus_synthetiques", "core", "chatbot / agent conversationnel",
            rf"\b(?:chatbots?|agents? conversationnels?|robots? conversationnels?)\b"),
    # --- Systèmes automatisés / autonomes -----------------------------------
    Keyword("automatisation", "core", "conduite automatisée / véhicule automatisé",
            rf"\b(?:equipes? d'un systeme d'automatisation|conduite automatisee|vehicules? automatise{_E}|vehicules? autonomes?|pilotage automatique)\b"),
    Keyword("automatisation", "core", "armes autonomes", rf"\b(?:systemes? d'armes? autonomes?|armes? autonomes?)\b"),
    # générique : inclut l'automatisation du bâtiment, de processus, etc.
    Keyword("automatisation", "extended", "système d'automatisation / système autonome",
            rf"\b(?:systemes? d'automatisation|systemes? autonomes?)\b"),
    Keyword("automatisation", "extended", "robot / robotique", rf"\b(?:robots?|robotique{_S}|robotise{_E})\b"),
    # --- Décision automatisée / profilage -----------------------------------
    Keyword("decision_automatisee", "core", "décision individuelle automatisée",
            rf"\b(?:decisions? (?:individuelles? )?automatisee{_S}|decisions? (?:prises? )?(?:de maniere|de facon) automatisee|decision fondee exclusivement sur un traitement automatise)\b"),
    Keyword("decision_automatisee", "extended", "profilage", rf"\bprofilages?\b"),
    Keyword("decision_automatisee", "extended", "traitement / procédure automatisé(e)",
            rf"\b(?:traitements? automatise{_E}|procedures? automatisee{_S}|systemes? automatise{_E}|de maniere automatisee|de facon automatisee|automatiquement (?:par|au moyen))\b"),
    Keyword("decision_automatisee", "extended", "aide à la décision", rf"\b(?:aide a la decision|outils? d'aide a la decision)\b"),
    # --- Algorithmes ---------------------------------------------------------
    Keyword("algorithmes", "extended", "algorithme", rf"\balgorithm(?:e{_S}|ique{_S})\b"),
    # --- Biométrie / reconnaissance -----------------------------------------
    Keyword("biometrie", "extended", "reconnaissance faciale / biométrique",
            rf"\b(?:reconnaissance (?:faciale|biometrique|des visages)|identification biometrique|comparaison biometrique|systemes? de reconnaissance (?:faciale|des visages|biometrique))\b"),
    Keyword("biometrie", "extended", "recherche automatisée de véhicules (plaques)",
            rf"\b(?:recherche automatisee de vehicules|lecture automatique des plaques|reconnaissance automatique des plaques)\b"),
    # --- Véhicules et engins sans conducteur ---------------------------------
    Keyword("automatisation", "extended", "véhicule / engin sans conducteur",
            rf"\b(?:vehicules?|trains?|navires?|bateaux?|aeronefs?) sans (?:conducteur|pilote|equipage)\b"),
    Keyword("automatisation", "extended", "drone", rf"\bdrones?\b"),
    # --- Technologies émergentes et espace informationnel (ajout pour les arrêtés fédéraux,
    #     qui parlent souvent de technologies sans nommer l'IA ; tri par l'étape LLM 1) ------
    Keyword("technologies_emergentes", "extended", "technologies émergentes / de rupture",
            rf"\btechnolog(?:ie|ies|ique|iques) (?:numeriques? )?(?:emergentes?|de rupture|disruptives?)\b|\bavancees technologiques\b"),
    Keyword("technologies_emergentes", "extended", "espace / intégrité cognitive",
            rf"\b(?:espace|integrite|guerre|domaine|resilience|securite) cognitive?s?\b"),
    Keyword("technologies_emergentes", "extended", "désinformation / opérations d'influence",
            rf"\b(?:desinformation|mesinformation|manipulation de l'information|operations? d'influence|campagnes? d'influence)\b"),
    Keyword("technologies_emergentes", "extended", "transformation numérique", rf"\btransformation numerique\b"),
    Keyword("technologies_emergentes", "extended", "systèmes intelligents / objets connectés",
            rf"\b(?:systemes?|logiciels?|outils?|acheminement|technologies?|applications?) intelligente?s?\b|\binternet des objets\b|\bobjets? connectes?\b"),
    # --- Calcul, prédiction, reconnaissance (ajouts après l'exploration du corpus complet) ----
    Keyword("calcul", "extended", "calcul haute performance / processeur graphique / quantique",
            rf"\b(?:calcul (?:a |de )?haute performance|calcul accelere|processeurs? graphiques?|puissance de calcul|capacites? de calcul|informatique quantique|ordinateurs? quantiques?)\b"),
    Keyword("decision_automatisee", "extended", "modèle de prédiction",
            rf"\b(?:modeles? de prediction|modeles? predictifs?)\b"),
    Keyword("biometrie", "extended", "reconnaissance optique de caractères",
            rf"\breconnaissance optique de caracteres\b"),
    Keyword("automatisation", "extended", "systèmes d'assistance à la conduite",
            rf"\b(?:systemes? d'(?:aide|assistance) a la conduite|assistance a la conduite)\b"),
    # --- Données (exploitation à grande échelle) ----------------------------
    Keyword("donnees", "extended", "fouille de textes et de données / mégadonnées",
            rf"\b(?:fouille de (?:textes et de )?donnees|exploration de textes et de donnees|text and data mining|megadonnees|big data|science des donnees)\b"),
]


# ============================================================
# Normalisation
# ============================================================

_APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", " ": " ", " ": " "})


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize(s: str, *, keep_case: bool = False) -> str:
    s = _strip_accents((s or "").translate(_APOS))
    s = " ".join(s.split())
    return s if keep_case else s.casefold()


def normalize_with_map(s: str, *, keep_case: bool = False):
    """Comme normalize(), mais renvoie aussi, pour chaque caractère du texte normalisé,
    l'indice du caractère d'origine correspondant (pour retrouver la forme exacte d'un
    terme dans l'article : accents, majuscules, apostrophes typographiques)."""
    out: List[str] = []
    idx: List[int] = []
    prev_space = True
    for i, ch in enumerate((s or "")):
        c = ch.translate(_APOS)
        if c.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
            continue
        prev_space = False
        c = _strip_accents(c)
        if not keep_case:
            c = c.casefold()
        for cc in c:
            out.append(cc)
            idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


# ============================================================
# Recherche
# ============================================================

@dataclass(frozen=True)
class _Compiled:
    kw: Keyword
    rx: re.Pattern


def compile_keywords(
    keywords: Iterable[Keyword] = KEYWORDS,
    tiers: Optional[Iterable[str]] = None,
    groups: Optional[Iterable[str]] = None,
) -> List[_Compiled]:
    tiers = set(tiers) if tiers else None
    groups = set(groups) if groups else None
    out = []
    for kw in keywords:
        if tiers and kw.tier not in tiers:
            continue
        if groups and kw.group not in groups:
            continue
        out.append(_Compiled(kw, re.compile(kw.pattern)))
    if not out:
        raise ValueError("Aucun mot-clé sélectionné (vérifier --tiers / --groups).")
    return out


def find_in_text(text: str, compiled: List[_Compiled]) -> List[Keyword]:
    if not text:
        return []
    low = normalize(text)
    cased = None
    hits = []
    for c in compiled:
        if c.kw.case_sensitive:
            cased = cased if cased is not None else normalize(text, keep_case=True)
            if c.rx.search(cased):
                hits.append(c.kw)
        elif c.rx.search(low):
            hits.append(c.kw)
    return hits


def filter_dataframe(
    df: pd.DataFrame,
    compiled: List[_Compiled],
    *,
    levels: Iterable[int] = (1, 2, 3, 4, 5, 6),
    id_col: str = "law_id",
) -> Dict[str, pd.DataFrame]:
    """Retourne {'rows': lignes des actes retenus, 'acts': résumé par acte, 'keywords': comptages}."""
    levels = set(int(x) for x in levels)
    lvl = pd.to_numeric(df["level"], errors="coerce")
    searchable = lvl.isin(levels)

    texts = (df["label"].fillna("").astype(str) + " \n " + df["text"].fillna("").astype(str))

    kw_labels: List[str] = [""] * len(df)
    kw_groups: List[str] = [""] * len(df)
    kw_tiers: List[str] = [""] * len(df)
    hit = [False] * len(df)
    for i, (is_s, t) in enumerate(zip(searchable.to_numpy(), texts.to_numpy())):
        if not is_s:
            continue
        found = find_in_text(t, compiled)
        if found:
            hit[i] = True
            kw_labels[i] = "|".join(dict.fromkeys(k.label for k in found))
            kw_groups[i] = "|".join(dict.fromkeys(k.group for k in found))
            kw_tiers[i] = "core" if any(k.tier == "core" for k in found) else "extended"

    df = df.copy()
    df["kw_hit"] = hit
    df["kw_keywords"] = kw_labels
    df["kw_groups"] = kw_groups
    df["kw_tier"] = kw_tiers

    hit_rows = df[df["kw_hit"]]
    if hit_rows.empty:
        empty = df.iloc[0:0]
        return {"rows": empty, "acts": pd.DataFrame(), "keywords": pd.DataFrame()}

    def _join(series: pd.Series) -> str:
        vals = []
        for v in series:
            vals.extend(x for x in v.split("|") if x)
        return "|".join(sorted(set(vals)))

    acts = (
        hit_rows.groupby(id_col)
        .agg(
            act_n_rows_hit=("kw_hit", "size"),
            act_keywords=("kw_keywords", _join),
            act_groups=("kw_groups", _join),
            act_has_core=("kw_tier", lambda s: bool((s == "core").any())),
        )
        .reset_index()
    )

    # métadonnées de l'acte (titre = ligne de niveau 0)
    meta_cols = [c for c in ["sr_number", "doc_category", "type_code", "doc_type_label"] if c in df.columns]
    titles = df[lvl == 0].drop_duplicates(id_col).set_index(id_col)
    meta = titles[meta_cols].copy()
    meta["title"] = titles["label"]
    acts = acts.join(meta, on=id_col)
    acts["act_tier"] = acts["act_has_core"].map({True: "core", False: "extended"})
    acts = acts.drop(columns=["act_has_core"])
    acts = acts[[id_col] + meta_cols + ["title", "act_tier", "act_n_rows_hit", "act_groups", "act_keywords"]]
    acts = acts.sort_values(["act_tier", "act_n_rows_hit"], ascending=[True, False]).reset_index(drop=True)

    rows = df[df[id_col].isin(acts[id_col])].merge(
        acts[[id_col, "act_tier", "act_groups", "act_keywords"]], on=id_col, how="left"
    )

    # comptages par mot-clé : nb de lignes et nb d'actes
    exploded = hit_rows[[id_col, "kw_keywords"]].assign(keyword=hit_rows["kw_keywords"].str.split("|")).explode("keyword")
    kw_counts = (
        exploded.groupby("keyword")
        .agg(n_rows=(id_col, "size"), n_acts=(id_col, "nunique"))
        .reset_index()
    )
    info = pd.DataFrame([{"keyword": c.kw.label, "group": c.kw.group, "tier": c.kw.tier, "pattern": c.kw.pattern} for c in compiled])
    kw_counts = info.merge(kw_counts, on="keyword", how="left").fillna({"n_rows": 0, "n_acts": 0})
    kw_counts[["n_rows", "n_acts"]] = kw_counts[["n_rows", "n_acts"]].astype(int)

    return {"rows": rows, "acts": acts, "keywords": kw_counts}


def find_surface_terms(text: str, compiled: List[_Compiled]) -> List[Dict[str, str]]:
    """Occurrences des mots-clés dans `text`, avec la forme EXACTE telle qu'écrite dans
    l'article (ex. 'décisions individuelles automatisées'). Une entrée par forme distincte ;
    un terme inclus dans un terme plus long trouvé au même endroit est omis."""
    if not text:
        return []
    low, low_map = normalize_with_map(text)
    cased, cased_map = None, None
    spans = []  # (start, end, keyword)
    for c in compiled:
        if c.kw.case_sensitive:
            if cased is None:
                cased, cased_map = normalize_with_map(text, keep_case=True)
            hay, m_ = cased, cased_map
        else:
            hay, m_ = low, low_map
        for m in c.rx.finditer(hay):
            if m.end() <= m.start():
                continue
            spans.append((m_[m.start()], m_[m.end() - 1] + 1, c.kw))
    # retirer les occurrences contenues dans une occurrence plus longue
    kept = [sp for sp in spans
            if not any(o is not sp and o[0] <= sp[0] and sp[1] <= o[1] and (o[1] - o[0]) > (sp[1] - sp[0]) for o in spans)]
    found: Dict[str, Dict[str, str]] = {}
    for a, b, kw in sorted(kept, key=lambda x: x[0]):
        surface = " ".join(text[a:b].split())
        surface = re.sub(r"^(?:[ldLD][’'])(?=IA\b)", "", surface)   # "l'IA" -> "IA"
        key = surface.casefold()
        if key not in found:
            found[key] = {"term": surface, "keyword": kw.label, "group": kw.group, "tier": kw.tier}
    return list(found.values())
