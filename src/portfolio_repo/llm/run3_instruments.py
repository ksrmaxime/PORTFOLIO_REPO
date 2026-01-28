from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ajuste l'import selon où est ton client.
# Si tu as déjà un module stable: from portfolio_repo.llm.client import LocalLLMClient, LLMConfig
from portfolio_repo.llm.client import LocalLLMClient, LLMConfig  # type: ignore


# ----------------------------
# Taxonomie + exemples (valeurs autorisées)
# ----------------------------
# On garde les labels EXACTS comme valeurs de sortie (stables), mais on enrichit leur définition
# dans le prompt avec des exemples / indices.
INSTRUMENT_TAXONOMY: List[str] = [
    "Instruments réglementaires contraignants",
    "Instruments de contrôle et de gestion des risques",
    "Instruments informationnels",
    "Instruments de standardisation et de conformité technique",
    "Instruments institutionnels et de supervision",
    "Instruments économiques et de marché",
    "Instruments expérimentaux",
    "Instruments de soft law encadrée",
    "Instruments de capacité étatique",
]

NONE_VALUE = "NONE"


def taxonomy_with_examples() -> str:
    """
    Sert uniquement au prompt (pas aux validations).
    Le but est d'augmenter le rappel sans laisser le modèle inventer une nouvelle catégorie.
    """
    return """\
1) Instruments réglementaires contraignants
   - Exemples: obligation légale ("doit", "est tenu de"), interdiction ("il est interdit de"),
     autorisation/licence/permis/approbation ("autorisation", "agrément", "concession"),
     obligation d'enregistrement, sanctions/amendes, responsabilité civile/pénale spécifique,
     pouvoirs de contrainte (ordonner, retirer, suspendre).

2) Instruments de contrôle et de gestion des risques
   - Exemples: analyse/évaluation des risques, étude d'impact, mesures de sécurité obligatoires,
     monitoring/surveillance continue, notification d'incidents, tenue de journaux/logs,
     documentation technique obligatoire, traçabilité, audits/inspections obligatoires.

3) Instruments informationnels
   - Exemples: obligation d’informer (usagers, personnes concernées), transparence,
     obligation de reporting/rapport, publication, registre public, accès à l'information,
     exigences d’explicabilité/motivation (ex: "motiver", "communiquer les raisons").

4) Instruments de standardisation et de conformité technique
   - Exemples: renvoi à des normes/standards (ISO, IEC, ETSI, etc.), "état de la technique"
     si explicitement lié à une conformité technique, certification/label officiel,
     exigences techniques détaillées, procédures de conformité.

5) Instruments institutionnels et de supervision
   - Exemples: création/mandat d’une autorité de surveillance, organe de contrôle,
     mécanisme de plainte/recours, compétences d'enquête, coordination interinstitutionnelle,
     désignation d’une autorité compétente, contrôle parlementaire/administratif spécialisé.

6) Instruments économiques et de marché
   - Exemples: subventions/aides, financement/programmes, incitations fiscales,
     appels à projets, marchés publics/clauses contractuelles publiques,
     obligations d’assurance, tarification/redevances incitatives.

7) Instruments expérimentaux
   - Exemples: régimes pilotes, dérogations temporaires encadrées, sandbox,
     clauses d’expérimentation ("à titre d’essai", "projet pilote"), durée limitée.

8) Instruments de soft law encadrée
   - Exemples: lignes directrices officielles, recommandations, codes de conduite reconnus,
     co-régulation, bonnes pratiques formalisées par l’État sans obligation directe.

9) Instruments de capacité étatique
   - Exemples: création d’unités d’expertise, formation obligatoire/plan de formation,
     infrastructures publiques de contrôle, renforcement de compétences/ressources,
     centre de compétence, recrutement/qualification explicitement prévu.
"""


SYSTEM_PROMPT = """Tu es un assistant de codage juridique très strict.

Tâche:
À partir du texte d’un article de loi et d’une liste ordonnée de "targets"
(ex: ["Development & Adoption","Data"]), identifier POUR CHAQUE target l’instrument
de politique publique le plus plausible utilisé dans l’article.

Règles STRICTES:
- Retourner EXACTEMENT un instrument par target, dans le MÊME ordre.
- Choisir UNIQUEMENT parmi les valeurs autorisées dans la taxonomie (ou "NONE").
- "NONE" seulement si aucun instrument n'est identifiable à partir du texte.
- Ne pas déduire par spéculation sectorielle.
- Donner une justification courte (indices textuels) par target.
- Ne pas citer de longues phrases.

Sortie:
JSON strict uniquement:
{
  "instruments": [...],
  "justifications": [...]
}
Les deux listes doivent avoir la même longueur que "targets".
"""


USER_TEMPLATE = """TAXONOMIE (valeurs autorisées; choisir exactement une valeur ou "NONE"):
{taxonomy_examples}

VALEURS AUTORISÉES (rappel; sortie doit être exactement une de ces valeurs ou "NONE"):
- {allowed_values}

TARGETS (liste ordonnée; tu dois aligner les instruments dans le même ordre):
{targets_json}

TEXTE ARTICLE:
{text}
"""


# ----------------------------
# Parsing / validation
# ----------------------------
def parse_targets(raw: Any) -> Optional[List[str]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, list):
        return [str(x) for x in raw]

    s = str(raw).strip()
    if not s:
        return None

    if s.startswith("[") and s.endswith("]"):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:
            pass

        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        cleaned: List[str] = []
        for p in parts:
            p = p.strip().strip("'").strip('"')
            if p:
                cleaned.append(p)
        return cleaned or None

    return [s]


def clean_text_for_llm(text: str, max_chars: int = 12000) -> str:
    t = re.sub(r"\s+\n", "\n", text)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + "\n\n[TRONQUÉ]"
    return t


def safe_json_load(s: str) -> Dict[str, Any]:
    s2 = s.strip()
    try:
        return json.loads(s2)
    except Exception:
        pass

    m = re.search(r"\{.*\}", s2, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Réponse non-JSON: {s2[:400]}")
    return json.loads(m.group(0))


def validate_alignment(targets: List[str], instruments: Any, justifications: Any) -> Tuple[List[str], List[str]]:
    if not isinstance(instruments, list) or not isinstance(justifications, list):
        raise ValueError("Champs instruments/justifications doivent être des listes.")
    if len(instruments) != len(targets) or len(justifications) != len(targets):
        raise ValueError(
            f"Alignement invalide: len(targets)={len(targets)}, "
            f"len(instruments)={len(instruments)}, len(justifications)={len(justifications)}"
        )

    out_instr: List[str] = []
    out_just: List[str] = []

    allowed = set(INSTRUMENT_TAXONOMY)

    for instr, jus in zip(instruments, justifications):
        instr_s = str(instr).strip() if instr is not None else ""
        jus_s = str(jus).strip() if jus is not None else ""

        if not instr_s:
            instr_s = NONE_VALUE

        # garde-fou conservateur
        if instr_s != NONE_VALUE and instr_s not in allowed:
            instr_s = NONE_VALUE

        out_instr.append(instr_s)
        out_just.append(jus_s)

    return out_instr, out_just


# ----------------------------
# LLM call (un article)
# ----------------------------
def extract_instruments_for_article(
    client: LocalLLMClient,
    targets: List[str],
    article_text: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[List[str], List[str], str]:
    text = clean_text_for_llm(article_text)
    user_prompt = USER_TEMPLATE.format(
        taxonomy_examples=taxonomy_with_examples(),
        allowed_values="\n- ".join(INSTRUMENT_TAXONOMY),
        targets_json=json.dumps(targets, ensure_ascii=False),
        text=text,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw = client.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    data = safe_json_load(raw)
    instr_list, just_list = validate_alignment(targets, data.get("instruments"), data.get("justifications"))
    return instr_list, just_list, raw


# ----------------------------
# Run config + dataframe processing
# ----------------------------
@dataclass(frozen=True)
class Run3Config:
    input_path: Path
    output_path: Path
    checkpoint_every: int = 50
    sleep_seconds: float = 0.0
    temperature: float = 0.0
    max_tokens: int = 600
    only_missing: bool = True


def run3_extract_instruments(cfg: Run3Config, llm_cfg: LLMConfig) -> int:
    df = pd.read_parquet(cfg.input_path)

    # Colonnes de sortie en fin (stables)
    if "article_instruments" not in df.columns:
        df["article_instruments"] = pd.NA
    if "article_instruments_justification" not in df.columns:
        df["article_instruments_justification"] = pd.NA

    mask = (df["level"] == 5) & df["article_targets"].notna()
    if cfg.only_missing:
        mask = mask & df["article_instruments"].isna()

    to_process = df.loc[mask]
    if to_process.empty:
        df.to_parquet(cfg.output_path, index=False)
        print("Aucun article à traiter (filtre vide).")
        return 0

    client = LocalLLMClient(llm_cfg)

    processed = 0
    errors = 0

    for idx, row in to_process.iterrows():
        targets = parse_targets(row["article_targets"])
        if not targets:
            continue

        text = str(row.get("text") or "").strip()
        if not text:
            df.at[idx, "article_instruments"] = json.dumps([NONE_VALUE] * len(targets), ensure_ascii=False)
            df.at[idx, "article_instruments_justification"] = json.dumps([""] * len(targets), ensure_ascii=False)
            processed += 1
            continue

        try:
            instr_list, just_list, _raw = extract_instruments_for_article(
                client=client,
                targets=targets,
                article_text=text,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            df.at[idx, "article_instruments"] = json.dumps(instr_list, ensure_ascii=False)
            df.at[idx, "article_instruments_justification"] = json.dumps(just_list, ensure_ascii=False)
            processed += 1
        except Exception as e:
            errors += 1
            df.at[idx, "article_instruments"] = json.dumps([NONE_VALUE] * len(targets), ensure_ascii=False)
            df.at[idx, "article_instruments_justification"] = json.dumps([f"ERROR: {type(e).__name__}"] * len(targets), ensure_ascii=False)

        if cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)

        if processed > 0 and processed % cfg.checkpoint_every == 0:
            df.to_parquet(cfg.output_path, index=False)
            print(f"Checkpoint: processed={processed}, errors={errors}, saved={cfg.output_path}")

    df.to_parquet(cfg.output_path, index=False)
    print(f"Terminé: processed={processed}, errors={errors}, saved={cfg.output_path}")
    return 0
