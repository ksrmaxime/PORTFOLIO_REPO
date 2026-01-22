from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import pandas as pd

from portfolio_repo.paths import ensure_dir
from portfolio_repo.fedlex.akn_clean_extract import ExtractConfig, extract_normative_structure


@dataclass(frozen=True)
class CleanBuildConfig:
    xml_glob: str = "*.xml"
    out_dir_rel: Tuple[str, ...] = ("processed", "fedlex")
    laws_parquet_name: str = "laws_clean.parquet"
    articles_parquet_name: str = "articles_clean.parquet"
    include_title: bool = True


def build_clean_parquets_from_xml_dir(
    xml_dir: str | Path,
    law_id_from_filename: bool = True,
    cfg: CleanBuildConfig = CleanBuildConfig(),
    extract_cfg: ExtractConfig = ExtractConfig(),
) -> Tuple[Path, Path]:
    """
    Reads all XMLs in a directory, builds:
      - laws_clean.parquet: one row per law with clean_text (full law)
      - articles_clean.parquet: one row per article with clean_text_article
    """
    xml_dir = Path(xml_dir)
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory does not exist: {xml_dir}")

    xml_paths = sorted(xml_dir.glob(cfg.xml_glob))
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found in {xml_dir} matching {cfg.xml_glob}")

    laws_rows: List[Dict[str, Any]] = []
    art_rows: List[Dict[str, Any]] = []

    for xp in xml_paths:
        law_rec, arts = extract_normative_structure(xp, cfg=extract_cfg)

        # derive law_id
        if law_id_from_filename:
            # your sample looks like "RS-235.1-07072025-FR.xml"
            # store full stem; you can later map it to your registry law_id if needed
            law_id = xp.stem
        else:
            law_id = law_rec.get("eli_uri", "") or xp.stem

        law_rec["law_id"] = law_id
        laws_rows.append(law_rec)

        for a in arts:
            a["law_id"] = law_id
            a["eli_uri"] = law_rec.get("eli_uri", "")
            a["lang"] = law_rec.get("lang", "")
            a["title"] = law_rec.get("title", "")
            art_rows.append(a)

    laws_df = pd.DataFrame(laws_rows)
    arts_df = pd.DataFrame(art_rows)

    out_dir = ensure_dir(Path.cwd() / "data" / Path(*cfg.out_dir_rel))
    laws_path = out_dir / cfg.laws_parquet_name
    arts_path = out_dir / cfg.articles_parquet_name

    laws_df.to_parquet(laws_path, index=False)
    arts_df.to_parquet(arts_path, index=False)

    return laws_path, arts_path
