from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "apertus-local"
    timeout: int = 120


class LocalLLMClient:
    """
    Thin client for a local OpenAI-compatible LLM server (llama-server).
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.chat_url = f"{config.base_url}/v1/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        r = requests.post(
            self.chat_url,
            json=payload,
            timeout=self.config.timeout,
        )
        r.raise_for_status()

        data = r.json()
        return data["choices"][0]["message"]["content"]
