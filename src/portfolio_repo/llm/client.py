from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import requests

# Hard-disable meta device init (important on clusters + accelerate)
os.environ.setdefault("ACCELERATE_USE_META_DEVICE", "0")


# =====================
# Generic LLM interface
# =====================
class LLMClient(Protocol):
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str: ...


# =========================
# OpenAI-compatible backend
# =========================
@dataclass
class OpenAIHTTPConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "apertus-local"
    timeout: int = 120


class OpenAIHTTPClient:
    def __init__(self, config: OpenAIHTTPConfig):
        self.config = config
        self.chat_url = f"{config.base_url.rstrip('/')}/v1/chat/completions"

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
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if response_format is not None:
            payload["response_format"] = response_format

        r = requests.post(self.chat_url, json=payload, timeout=self.config.timeout)
        if not r.ok:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        return r.json()["choices"][0]["message"]["content"]


# Backward compatibility (old imports)
@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "apertus-local"
    timeout: int = 120


class LocalLLMClient(OpenAIHTTPClient):
    def __init__(self, config: LLMConfig):
        super().__init__(
            OpenAIHTTPConfig(
                base_url=config.base_url,
                model=config.model,
                timeout=config.timeout,
            )
        )


# =========================
# Transformers local backend
# =========================
@dataclass
class TransformersConfig:
    model_path: str
    device: str = "auto"     # auto | cpu | cuda
    dtype: str = "bf16"      # bf16 | fp16 | auto
    trust_remote_code: bool = True


class TransformersClient:
    """
    Local inference via transformers.

    Apertus + transformers 5 can trigger meta-tensors in init.
    We hard-disable meta init via ACCELERATE_USE_META_DEVICE=0,
    and load on CPU first (no device_map), then move to GPU.
    """

    def __init__(self, config: TransformersConfig):
        self.config = config

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
        )

        # device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        # dtype (use transformers>=5 style: dtype=...)
        if config.dtype == "bf16":
            dtype = torch.bfloat16
        elif config.dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = "auto"

        # Ensure default device is CPU while constructing (avoid meta / weird defaults)
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

        # model load: NO device_map, NO low_cpu_mem_usage
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
            dtype=dtype,                 # <-- critical (no torch_dtype)
            device_map=None,
            low_cpu_mem_usage=False,
        )

        self.model.eval()
        self.model.to(self.device)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = float(temperature) > 0.0

        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):].lstrip()
        return text
