# src/portfolio_repo/fedlex/chunking.py
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List

from portfolio_repo.fedlex.cleaning import CleaningConfig, clean_fedlex_text


def xml_to_plain_text(
    xml_bytes: bytes,
    *,
    apply_cleaning: bool = True,
    cleaning_cfg: CleaningConfig | None = None,
) -> str:
    """
    Best-effort conversion of Fedlex Akoma Ntoso / ELI XML into plain text.

    Default behavior: apply Fedlex-specific cleaning to remove editorial noise.
    Set apply_cleaning=False if you want the raw extracted text.
    """
    root = ET.fromstring(xml_bytes)

    parts: List[str] = []
    for t in root.itertext():
        if t:
            parts.append(t)

    # Preserve line breaks if present, normalize spaces.
    text = "".join(parts)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if apply_cleaning:
        text = clean_fedlex_text(text, cfg=cleaning_cfg)

    return text


@dataclass(frozen=True)
class Chunk:
    unit_id: str
    chunk_index: int
    text: str


def chunk_text(text: str, target_chars: int = 10000, overlap_chars: int = 300) -> List[Chunk]:
    """
    Simple, stable chunking for law-level triage.
    - target_chars: approximate size of each chunk
    - overlap_chars: small overlap to preserve continuity
    """
    if target_chars < 1000:
        raise ValueError("target_chars too small (min 1000)")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be < target_chars")

    n = len(text)
    if n == 0:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 1
    while start < n:
        end = min(n, start + target_chars)
        chunk = text[start:end]
        chunks.append(Chunk(unit_id=f"chunk_{idx:03d}", chunk_index=idx, text=chunk))
        if end == n:
            break
        start = max(0, end - overlap_chars)
        idx += 1

    return chunks
