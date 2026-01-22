from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CleaningConfig:
    """
    Conservative cleaning for Fedlex consolidated texts (plain text already extracted from XML).

    Goal: remove *editorial / consolidation apparatus* (revision notes, RO/FF blocks, "État le ...",
    "Nouvelle teneur...", etc.) while keeping normative content.

    This stays "pipeline-friendly": same signature, same style (line-based), but stronger patterns and
    a few safe multi-line block rules.
    """
    drop_reference_lines: bool = True
    drop_revision_lines: bool = True
    drop_editorial_blocks: bool = True
    drop_inline_refs: bool = True
    drop_footnote_markers: bool = True
    normalize_whitespace: bool = True
    collapse_blank_lines: bool = True

    # If True, removes "mostly-reference" lines even if they contain a bit of text
    drop_reference_heavy_lines: bool = True

    # If True, deletes short lines that are only punctuation / separators
    drop_separator_lines: bool = True


# -----------------------------
# Single-line patterns
# -----------------------------

# Common header fragments in consolidations
_ETAT_LE_RE = re.compile(r"\b(?:État|Etat)\s+le\s+\d{1,2}\s+\w+\s+\d{4}\b", re.IGNORECASE)

# Lines that are almost always editorial annotations in consolidations (FR/DE/IT mix, but conservative)
_REVISION_LINE_RE = re.compile(
    r"""^\s*(?:
        Nouvelle\s+teneur\b|
        Ancienne\s+teneur\b|
        Introduit(?:e)?\s+par\b|
        Abrog(?:é|ée)\b|
        Remplac(?:é|ée)\b|
        Modifi(?:é|ée)\b|
        Selon\s+le\s+ch\.\b|
        Actuellement\b|
        Disposition\s+transitoire\b|
        Dispositions?\s+finales?\b|
        Entrée?\s+en\s+vigueur\b|
        Version\b|
        Note\s+de\s+bas\s+de\s+page\b
    ).*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Standalone references often appear as their own lines (keep conservative about RS)
_REFERENCE_LINE_RE = re.compile(
    r"""^\s*(?:
        (?:RO|AS)\s+\d{4}.*|
        FF\s+\d{4}.*|
        \(\s*(?:RO|AS|FF)\s+\d{4}.*\)\s*$
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Heuristic: lines dominated by citations / references (RO/AS/FF/RS + numbers), typically non-normative
_REFERENCE_HEAVY_RE = re.compile(
    r"""(?ix)
    \b(?:RO|AS|FF|RS)\b|
    \b\d{4}\b|
    \b\d{1,4}\s*(?:\.\d+)+\b
    """
)

# Separator lines like "----" or "*****"
_SEPARATOR_LINE_RE = re.compile(r"^\s*[-–—•*_=]{3,}\s*$")

# Inline parenthetical editorial references: "( RO 2018 3269 ; FF 2017 5837 )" / "(AS 2021 1234)"
_INLINE_RO_FF_PAREN_RE = re.compile(r"\(\s*(?:RO|AS|FF)\s+\d{4}[^)]*\)", re.IGNORECASE)

# Inline RO/AS/FF fragments not necessarily in parentheses
_INLINE_RO_FF_RE = re.compile(r"\b(?:RO|AS|FF)\s+\d{4}\s+\d+\b", re.IGNORECASE)

# Footnote markers commonly produced by exporters: "1", "1)", "[1]", "(1)" at end of line
_FOOTNOTE_MARKER_EOF_RE = re.compile(r"(?:\s*(?:\[\d+\]|\(\d+\)|\d+\))\s*)$")

# In-text footnote markers: "¹", "²", etc. (superscripts)
_SUPERSCRIPT_DIGITS_RE = re.compile(r"[\u00B9\u00B2\u00B3\u2070-\u2079]+")


# -----------------------------
# Multi-line editorial blocks
# -----------------------------
# These appear as section headers followed by mostly citations / amendment prose.
# We remove the whole block until a clear structural boundary (blank line + next "Art." / "Chap." etc.)
_BLOCK_START_RE = re.compile(
    r"""(?ix)^\s*(?:
        Modifications?\b|
        Historique\b|
        Notes?\b|
        Renvois?\b|
        Références?\b|
        Sources?\b|
        Bibliographie\b
    )\s*$"""
)

# A new normative unit boundary: new article / chapter / section (FR/DE/IT variants, conservative)
_NORM_BOUNDARY_RE = re.compile(
    r"""(?ix)^\s*(?:
        Art\.?\s+\d+[a-zA-Z]*\b|
        Chap(?:itre)?\.?\s+\w+|
        Section\s+\w+|
        Titre\s+\w+|
        Abschnitt\s+\w+|
        Kapitel\s+\w+|
        Sezione\s+\w+|
        Titolo\s+\w+
    )"""
)


def _is_reference_heavy(line: str) -> bool:
    """
    Heuristic: if a line contains multiple reference tokens and little else, it's likely editorial.
    """
    if not line.strip():
        return False
    hits = len(_REFERENCE_HEAVY_RE.findall(line))
    # length-normalized: allow some refs inside normative lines, but drop if line is mostly refs
    # Example: "RO 2018 1234; FF 2017 5678" => hits high, content low
    alpha = sum(ch.isalpha() for ch in line)
    digits = sum(ch.isdigit() for ch in line)
    # Conservative thresholds
    return (hits >= 3 and alpha < 25) or (digits > alpha * 2 and hits >= 2)


def clean_fedlex_text(text: str, cfg: CleaningConfig | None = None) -> str:
    cfg = cfg or CleaningConfig()

    lines = text.splitlines()
    out_lines: list[str] = []

    in_editorial_block = False

    for ln in lines:
        raw = ln.rstrip()

        if cfg.drop_separator_lines and _SEPARATOR_LINE_RE.match(raw):
            continue

        # Normalize/remove header fragments (keeps rest of line)
        raw = _ETAT_LE_RE.sub("", raw).strip()

        # Detect start/end of editorial blocks
        if cfg.drop_editorial_blocks:
            if not in_editorial_block and _BLOCK_START_RE.match(raw):
                in_editorial_block = True
                continue

            if in_editorial_block:
                # End block if we hit a strong boundary (new article/chapter) OR two blank lines in a row
                if _NORM_BOUNDARY_RE.match(raw):
                    in_editorial_block = False
                    # do not skip this boundary line; let it be processed normally below
                else:
                    # stay in block; skip everything (including blank lines) until boundary
                    continue

        # Drop typical revision/annotation lines
        if cfg.drop_revision_lines and _REVISION_LINE_RE.match(raw):
            continue

        # Drop standalone reference lines
        if cfg.drop_reference_lines and _REFERENCE_LINE_RE.match(raw):
            continue

        # Drop reference-heavy lines (heuristic)
        if cfg.drop_reference_heavy_lines and _is_reference_heavy(raw):
            continue

        # Inline cleaning
        if cfg.drop_inline_refs:
            raw = _INLINE_RO_FF_PAREN_RE.sub("", raw)
            raw = _INLINE_RO_FF_RE.sub("", raw)

        if cfg.drop_footnote_markers:
            raw = _SUPERSCRIPT_DIGITS_RE.sub("", raw)
            raw = _FOOTNOTE_MARKER_EOF_RE.sub("", raw)

        out_lines.append(raw)

    cleaned = "\n".join(out_lines)

    if cfg.normalize_whitespace:
        # Keep newlines, but normalize spaces/tabs
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)

    if cfg.collapse_blank_lines:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned

