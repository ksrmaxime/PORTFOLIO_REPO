from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class SparqlConfig:
    endpoint: str = "https://fedlex.data.admin.ch/sparqlendpoint"
    timeout_s: int = 60
    max_retries: int = 4
    backoff_s: float = 1.5


class SparqlClient:
    def __init__(self, cfg: SparqlConfig | None = None) -> None:
        self.cfg = cfg or SparqlConfig()

    def query_json(self, sparql: str) -> Dict[str, Any]:
        headers = {"Accept": "application/sparql-results+json"}
        data = {"query": sparql}

        last_err: Optional[Exception] = None
        for i in range(self.cfg.max_retries):
            try:
                r = requests.post(self.cfg.endpoint, data=data, headers=headers, timeout=self.cfg.timeout_s)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff_s * (i + 1))

        raise RuntimeError(f"SPARQL query failed after retries: {last_err}") from last_err
