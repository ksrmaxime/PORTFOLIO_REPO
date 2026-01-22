# src/portfolio_repo/fedlex/download.py
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import requests


def safe_law_folder_name(eli_uri: str) -> str:
    """
    Make a stable folder name from an ELI URI:
    - keep last path segment (if any) for readability
    - add a short hash to prevent collisions
    """
    last = eli_uri.rstrip("/").split("/")[-1]
    last = re.sub(r"[^A-Za-z0-9._-]+", "_", last)[:80] or "law"
    h = hashlib.sha1(eli_uri.encode("utf-8")).hexdigest()[:12]
    return f"{last}__{h}"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class DownloaderConfig:
    timeout_s: int = 60
    max_retries: int = 3
    backoff_s: float = 1.5
    sleep_between_requests_s: float = 0.15


def fetch_xml_for_law(eli_uri: str, lang: str, cfg: DownloaderConfig) -> Tuple[bytes, int]:
    """
    Fetch XML representation using content negotiation.
    Many resources will respect:
      Accept: application/xml
      Accept-Language: de/fr/it
    """
    headers = {
        "Accept": "application/xml",
        "Accept-Language": lang,
        "User-Agent": "portfolio_repo/0.1 (research; contact: none)",
    }

    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        try:
            r = requests.get(eli_uri, headers=headers, timeout=cfg.timeout_s)
            status = r.status_code
            if status >= 500:
                raise RuntimeError(f"Server error {status}")
            r.raise_for_status()
            return r.content, status
        except Exception as e:
            last_err = e
            time.sleep(cfg.backoff_s * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {eli_uri} lang={lang}: {last_err}")


def save_xml(raw_dir: Path, eli_uri: str, lang: str, xml_bytes: bytes) -> Path:
    folder = raw_dir / "fedlex_xml" / safe_law_folder_name(eli_uri)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{lang}.xml"
    path.write_bytes(xml_bytes)
    return path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
