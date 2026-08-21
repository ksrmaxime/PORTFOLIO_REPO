# src/run6_config.py
from __future__ import annotations

from collections import OrderedDict

from src.run5_prompts import INSTRUMENT_CODES, TARGET_CODES

# Grouping of targets into the four top-level public-problem domains that
# define the columns of the portfolio matrix (PoC PDF, section 2.3 / Figure 1).
# Order within and across domains follows Figure 1 left-to-right.
TARGET_DOMAINS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("Data", ["PERSONAL_DATA", "IP_CREATIVE_CONTENT"]),
        ("Skills", ["EDUCATION", "RESEARCH"]),
        ("Infrastructure", ["COMPUTE_HARDWARE", "DATA_CENTERS_ENERGY"]),
        (
            "Risk & Societal Harms",
            ["HIGH_STAKES_APPS", "ALGORITHMIC_ACCOUNTABILITY", "DISINFORMATION", "CYBERSECURITY_AI"],
        ),
    ]
)

TARGET_ORDER: list[str] = [code for codes in TARGET_DOMAINS.values() for code in codes]
INSTRUMENT_ORDER: list[str] = list(INSTRUMENT_CODES.keys())

if set(TARGET_ORDER) != set(TARGET_CODES.keys()):
    raise RuntimeError("TARGET_DOMAINS in run6_config.py is out of sync with run5_prompts.TARGET_CODES")
