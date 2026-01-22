from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from portfolio_repo.fedlex.sparql import SparqlClient
from portfolio_repo.paths import data_dir, ensure_dir


@dataclass(frozen=True)
class DownloadConfig:
    timeout_s: int = 60
    user_agent: str = "portfolio_repo/1.0 (fedlex downloader)"
    overwrite: bool = False
    # Cache SPARQL lookups for manifestation -> file URL
    cache_manifestation_resolution: bool = True


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _looks_like_html(content: bytes) -> bool:
    head = content[:800].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:200]


def _looks_like_xml(content: bytes) -> bool:
    head = content.lstrip()[:100]
    return head.startswith(b"<") and not _looks_like_html(content)


def _manifestation_uri(consolidation_uri: str, lang: str = "fr") -> str:
    # consolidation_uri: https://fedlex.data.admin.ch/eli/cc/2022/491/20250707
    # manifestation:     https://fedlex.data.admin.ch/eli/cc/2022/491/20250707/fr/xml
    return f"{consolidation_uri.rstrip('/')}/{lang}/xml"


def _resolve_filestore_url(
    sparql: SparqlClient,
    manifestation: str,
    cache: Optional[Dict[str, Optional[str]]] = None,
) -> Optional[str]:
    """
    Resolve the concrete file URL behind a manifestation resource using jolux:isExemplifiedBy.
    Returns the filestore URL, or None if not found.
    """
    if cache is not None and manifestation in cache:
        return cache[manifestation]

    q = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>

SELECT ?file
WHERE {{
  <{manifestation}> jolux:isExemplifiedBy ?file .
}}
LIMIT 1
"""
    js = sparql.query_json(q)
    bindings = js.get("results", {}).get("bindings", [])
    if not bindings:
        if cache is not None:
            cache[manifestation] = None
        return None

    file_url = bindings[0].get("file", {}).get("value")
    if cache is not None:
        cache[manifestation] = file_url
    return file_url


def download_cc_xml_batch(catalog: pd.DataFrame, cfg: DownloadConfig) -> pd.DataFrame:
    raw_root = ensure_dir(data_dir("raw") / "fedlex_cc_xml" / "fr")

    required = {"base_act_uri", "consolidation_uri", "consolidation_date_yyyymmdd"}
    missing = required - set(catalog.columns)
    if missing:
        raise RuntimeError(f"Catalog missing required columns: {sorted(missing)}")

    headers = {
        "User-Agent": cfg.user_agent,
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
    }

    sparql = SparqlClient()
    cache: Optional[Dict[str, Optional[str]]] = {} if cfg.cache_manifestation_resolution else None

    log_rows: List[Dict[str, Any]] = []

    for _, row in tqdm(catalog.iterrows(), total=len(catalog)):
        base_act_uri = str(row["base_act_uri"])
        cons_uri = str(row["consolidation_uri"])
        cons_date = str(row["consolidation_date_yyyymmdd"])

        ident = base_act_uri.rstrip("/").split("/")[-1]
        out_path = raw_root / f"{cons_date}__{ident}.xml"

        # Skip existing
        if out_path.exists() and not cfg.overwrite:
            log_rows.append(
                dict(
                    base_act_uri=base_act_uri,
                    cons_date=cons_date,
                    manifestation_uri=_manifestation_uri(cons_uri, "fr"),
                    file_url=None,
                    ok=True,
                    skipped_existing=True,
                    path=str(out_path),
                    sha256=None,
                    http_status=None,
                    error=None,
                )
            )
            continue

        manifestation = _manifestation_uri(cons_uri, "fr")

        try:
            file_url = _resolve_filestore_url(sparql, manifestation, cache=cache)
            if not file_url:
                log_rows.append(
                    dict(
                        base_act_uri=base_act_uri,
                        cons_date=cons_date,
                        manifestation_uri=manifestation,
                        file_url=None,
                        ok=False,
                        skipped_existing=False,
                        path=None,
                        sha256=None,
                        http_status=None,
                        error="No jolux:isExemplifiedBy file URL for manifestation",
                    )
                )
                continue

            r = requests.get(file_url, headers=headers, timeout=cfg.timeout_s, allow_redirects=True)
            content = r.content
            ct = (r.headers.get("Content-Type") or "").lower()

            if r.status_code != 200:
                log_rows.append(
                    dict(
                        base_act_uri=base_act_uri,
                        cons_date=cons_date,
                        manifestation_uri=manifestation,
                        file_url=file_url,
                        ok=False,
                        skipped_existing=False,
                        path=None,
                        sha256=None,
                        http_status=r.status_code,
                        error=f"HTTP {r.status_code}",
                    )
                )
                continue

            if not _looks_like_xml(content):
                log_rows.append(
                    dict(
                        base_act_uri=base_act_uri,
                        cons_date=cons_date,
                        manifestation_uri=manifestation,
                        file_url=file_url,
                        ok=False,
                        skipped_existing=False,
                        path=None,
                        sha256=None,
                        http_status=r.status_code,
                        error=f"Non-XML content (ct={ct}, html={_looks_like_html(content)})",
                    )
                )
                continue

            out_path.write_bytes(content)

            log_rows.append(
                dict(
                    base_act_uri=base_act_uri,
                    cons_date=cons_date,
                    manifestation_uri=manifestation,
                    file_url=file_url,
                    ok=True,
                    skipped_existing=False,
                    path=str(out_path),
                    sha256=_sha256_bytes(content),
                    http_status=r.status_code,
                    error=None,
                )
            )

        except Exception as e:
            log_rows.append(
                dict(
                    base_act_uri=base_act_uri,
                    cons_date=cons_date,
                    manifestation_uri=manifestation,
                    file_url=None,
                    ok=False,
                    skipped_existing=False,
                    path=None,
                    sha256=None,
                    http_status=None,
                    error=str(e),
                )
            )

    return pd.DataFrame(log_rows)


