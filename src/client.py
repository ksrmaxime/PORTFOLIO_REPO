# src/client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os

os.environ.setdefault("ACCELERATE_USE_META_DEVICE", "0")


@dataclass(frozen=True)
class LLMConfig:
    model_path: str
    dtype: str = "bf16"          # "bf16" | "fp16"
    trust_remote_code: bool = True


class TransformersClient:
    def __init__(self, cfg: LLMConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
        )

        dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16

        # Évite certains soucis accelerate/meta device
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
        ).eval().to("cuda")

    def chat_many(
        self,
        system_prompt: str,
        user_prompts: List[str],
        temperature: float = 0.0,
        max_new_tokens: int = 200,
    ) -> List[str]:
        if not user_prompts:
            return []

        # 1) build chat prompts
        prompts: List[str] = []
        for up in user_prompts:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": up},
            ]
            prompts.append(
                self.tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        # 2) tokenize batch
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        do_sample = float(temperature) > 0.0

        # 3) generate
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
            )

        # 4) decode only generated part (robuste avec attention_mask)
        attn = enc.get("attention_mask")
        input_lens = attn.sum(dim=1).tolist() if attn is not None else [enc["input_ids"].shape[1]] * out.shape[0]

        res: List[str] = []
        for i in range(out.shape[0]):
            in_len = int(input_lens[i])
            gen_ids = out[i, in_len:]
            txt = self.tok.decode(gen_ids, skip_special_tokens=True)
            res.append(txt.strip())
        return res