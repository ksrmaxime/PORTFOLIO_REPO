# src/portfolio_repo/llm/curnagl_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import os

# Important on clusters with accelerate/transformers meta-init behavior
os.environ.setdefault("ACCELERATE_USE_META_DEVICE", "0")


@dataclass(frozen=True)
class TransformersConfig:
    model_path: str
    dtype: str = "bf16"   # bf16 | fp16
    trust_remote_code: bool = True


class TransformersClient:
    def __init__(self, cfg: TransformersConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
        )

        if cfg.dtype == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        # Build on CPU first, then move to GPU (stable on clusters)
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
            dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        self.model.eval()
        self.model.to("cuda")

    def chat_many(
        self,
        system_prompt: str,
        user_prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> List[str]:
        """
        True GPU batching: one generate() for many prompts.
        Returns one string per prompt (assistant completion only).
        Robustly removes the prompt by slicing generated token ids (not string prefix matching).
        """
        if not user_prompts:
            return []

        prompts: List[str] = []
        for up in user_prompts:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": up},
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = float(temperature) > 0.0

        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
            )

        # Compute per-row input lengths (excluding padding)
        # attention_mask is 1 for tokens, 0 for padding
        attn = inputs.get("attention_mask", None)
        if attn is None:
            # Fallback (should not happen with HF tokenizers)
            input_lens = [inputs["input_ids"].shape[1]] * out.shape[0]
        else:
            input_lens = attn.sum(dim=1).tolist()

        completions: List[str] = []
        for i in range(out.shape[0]):
            in_len = int(input_lens[i])
            gen_ids = out[i, in_len:]
            txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(txt.strip())

        return completions