# src/portfolio_repo/fedlex/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LawRecord:
    law_id: str            # stable internal id (we use the ELI URI by default)
    eli_uri: str
    sr_number: Optional[str]
    title: Optional[str]
    lang: Optional[str]    # for title if we retrieved a specific language


@dataclass(frozen=True)
class DownloadResult:
    law_id: str
    eli_uri: str
    lang: str
    xml_path: Optional[str]
    ok: bool
    http_status: Optional[int]
    error: Optional[str]
    sha256: Optional[str]
    downloaded_at_iso: Optional[str]
