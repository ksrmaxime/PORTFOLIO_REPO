from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List
import hashlib
import time

import pandas as pd
import requests

from portfolio_repo.paths import data_dir, ensure_dir


@dataclass(frozen=True)
class DownloadConfig:
    timeout_s: int = 60
    max_retries: int = 3
    backoff_s: float = 1.5
    user_agent: str = "portfolio-repo-fedlex-downloader/1.0"
    # Output
    raw_subdir: tuple[str, ...] = ("raw", "fedlex", "xml")  # flat folder
    log_subdir: tuple[str, ...] = ("raw", "fedlex", "logs")
    log_name: str = "download_xml_log.parquet"


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _safe_filename(s: str) -> str:
    # conservative: keep alnum, dash, underscore, dot
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def download_one_xml(
    *,
    xml_url: str,
    out_path: Path,
    cfg: DownloadConfig,
    session: requests.Session,
) -> Dict[str, Any]:
    """
    Download one XML to out_path. Writes atomically.
    Returns a log dict.
    """
    headers = {"User-Agent": cfg.user_agent}
    last_err: Optional[str] = None
    status: Optional[int] = None
    content_type: Optional[str] = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            r = session.get(xml_url, headers=headers, timeout=cfg.timeout_s)
            status = r.status_code
            content_type = r.headers.get("Content-Type")

            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                raise RuntimeError(last_err)

            data = r.content
            sha = _sha256_bytes(data)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(out_path)

            return {
                "ok": True,
                "http_status": status,
                "error": None,
                "content_type": content_type,
                "sha256": sha,
                "bytes": len(data),
            }

        except Exception as e:
            last_err = str(e)
            if attempt < cfg.max_retries:
                time.sleep(cfg.backoff_s * attempt)
            else:
                return {
                    "ok": False,
                    "http_status": status,
                    "error": last_err,
                    "content_type": content_type,
                    "sha256": None,
                    "bytes": None,
                }

    # unreachable
    return {"ok": False, "http_status": status, "error": last_err}


def download_many_xml_flat(
    records: pd.DataFrame,
    *,
    url_col: str = "xml_url",
    law_id_col: str = "law_id",
    lang_col: str = "lang",
    cfg: DownloadConfig = DownloadConfig(),
    overwrite: bool = False,
) -> Path:
    """
    Download all XML URLs listed in `records` into one flat folder: data/raw/fedlex/xml

    Expected columns:
      - law_id (string)
      - xml_url (string)
      - lang (string) optional but recommended

    Output filenames:
      - {law_id}__{lang}.xml  (lang omitted if missing)

    Writes/updates a parquet log at data/raw/fedlex/logs/download_xml_log.parquet
    """
    if url_col not in records.columns or law_id_col not in records.columns:
        raise ValueError(f"records must contain columns: {law_id_col}, {url_col}")

    out_dir = ensure_dir(data_dir(*cfg.raw_subdir))
    log_dir = ensure_dir(data_dir(*cfg.log_subdir))
    log_path = log_dir / cfg.log_name

    session = requests.Session()

    rows: List[Dict[str, Any]] = []
    for _, row in records.iterrows():
        law_id = str(row[law_id_col])
        xml_url = str(row[url_col])
        lang = str(row[lang_col]) if lang_col in records.columns and pd.notna(row[lang_col]) else ""

        base = _safe_filename(law_id)
        if lang:
            fname = f"{base}__{_safe_filename(lang)}.xml"
        else:
            fname = f"{base}.xml"

        out_path = out_dir / fname

        skipped_existing = False
        if out_path.exists() and not overwrite:
            skipped_existing = True
            rows.append(
                {
                    "law_id": law_id,
                    "lang": lang,
                    "xml_url": xml_url,
                    "xml_path": str(out_path),
                    "ok": True,
                    "http_status": None,
                    "error": None,
                    "content_type": None,
                    "sha256": _sha256_bytes(out_path.read_bytes()),
                    "bytes": out_path.stat().st_size,
                    "skipped_existing": True,
                }
            )
            continue

        res = download_one_xml(xml_url=xml_url, out_path=out_path, cfg=cfg, session=session)
        rows.append(
            {
                "law_id": law_id,
                "lang": lang,
                "xml_url": xml_url,
                "xml_path": str(out_path),
                "ok": bool(res["ok"]),
                "http_status": res.get("http_status"),
                "error": res.get("error"),
                "content_type": res.get("content_type"),
                "sha256": res.get("sha256"),
                "bytes": res.get("bytes"),
                "skipped_existing": skipped_existing,
            }
        )

    log_df = pd.DataFrame(rows)
    log_df.to_parquet(log_path, index=False)
    return log_path
