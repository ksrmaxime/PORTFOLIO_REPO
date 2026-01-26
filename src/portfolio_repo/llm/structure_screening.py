from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from portfolio_repo.llm.client import LocalLLMClient


@dataclass(frozen=True)
class StructureScreeningConfig:
    temperature: float = 0.0
    max_tokens: int = 700
    debug: bool = False
    debug_dir: str = "data/debug/structure_prompts"


def _build_prompt(law_id: int, labels: Dict[int, str]):
    # Liste courte, générique (pas spécifique “circulation”), mais suffisamment discriminante.
    # Le modèle doit se baser sur une occurrence EXPLICITE dans le libellé.
    keywords = [
        "automatis",              # automatisé, automatisation, automatique...
        "traitement automatis",
        "système automatis",
        "algorith",
        "intelligence artificielle",
        " IA", "IA ",             # petit hack pour éviter de matcher des sous-chaînes bizarres
        "donnée", "données", "data",
        "informatique",
        "numérique",
        "électronique", "electronique",
        "système d’information", "système d'information",
        "systèmes d’information", "systèmes d'information",
        "système d'information", "systeme d'information",
        "système d’information", "systeme d’information",
    ]

    system = (
        "Tu analyses une liste de TITRES juridiques appartenant à UNE seule loi.\n\n"
        "Tâche : sélectionner les titres en lien avec des systèmes automatisés/informatiques, "
        "le traitement de données, des algorithmes, ou l’IA.\n\n"
        "RÈGLE STRICTE (obligatoire) :\n"
        "- Tu n’as le droit de sélectionner un titre QUE SI le libellé contient explicitement "
        "au moins un mot/fragment de la LISTE DE MOTS-CLÉS fournie.\n"
        "- Pour chaque titre sélectionné, la justification doit contenir le ou les fragments EXACTS "
        "copiés depuis le libellé (entre guillemets). Pas de paraphrase.\n"
        "- Si tu ne peux pas citer un fragment exact présent dans le libellé, NE SÉLECTIONNE PAS.\n\n"
        "Retourne UNIQUEMENT du JSON strict."
    )

    lines = "\n".join(f"[{i}] {label}" for i, label in labels.items())

    user = (
        f"Loi ID: {law_id}\n\n"
        "LISTE DE MOTS-CLÉS (match simple dans le libellé ; casse/accents peuvent varier) :\n"
        + "\n".join(f"- {k}" for k in keywords) +
        "\n\n"
        "Titres juridiques :\n\n"
        f"{lines}\n\n"
        "Retour attendu (JSON strict) :\n"
        '{ "selected": [ { "id": 4, "reason": "\"mot exact du titre\"" } ] }\n\n'
        "Notes :\n"
        "- reason doit contenir au moins un extrait exact du libellé entre guillemets.\n"
        "- Si aucun titre ne correspond, retourne une liste vide."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def screen_law_structure(
    client: LocalLLMClient,
    law_numeric_id: int,
    structure_df: pd.DataFrame,
    cfg: StructureScreeningConfig,
) -> Dict[str, str]:
    """
    Analyse la structure d'une loi (labels uniquement).

    Retourne :
        dict { node_id (str) -> justification (str) }
    """

    g = structure_df[structure_df["node_type"] != "article"].copy()
    g = g.reset_index(drop=True)

    if g.empty:
        return {}

    g["local_id"] = g.index + 1

    labels: Dict[int, str] = {
        int(row.local_id): str(row.label).strip()
        for row in g.itertuples()
        if isinstance(row.label, str) and row.label.strip()
    }

    if not labels:
        return {}

    messages = _build_prompt(law_numeric_id, labels)

    if cfg.debug:
        Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)
        debug_path = Path(cfg.debug_dir) / f"law_{law_numeric_id}.txt"
        debug_content = []
        for m in messages:
            debug_content.append(f"[{m['role'].upper()}]\n{m['content']}\n")
        debug_path.write_text("\n".join(debug_content), encoding="utf-8")

    response = client.chat(
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return {}

    selected = data.get("selected", [])
    if not isinstance(selected, list):
        return {}

    node_reason: Dict[str, str] = {}

    for item in selected:
        if not isinstance(item, dict):
            continue

        local_id = item.get("id")
        reason = item.get("reason", "")

        if not isinstance(local_id, int):
            continue

        match = g[g["local_id"] == local_id]
        if match.empty:
            continue

        node_id = match["node_id"].iloc[0]
        node_reason[node_id] = str(reason).strip()

    return node_reason
