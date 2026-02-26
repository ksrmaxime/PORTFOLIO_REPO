# src/client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
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

        # Stabilise padding + évite warnings/bugs
        # (le slicing correct ci-dessous marche même sans ça, mais c'est plus propre)
        self.tok.padding_side = "right"
        if self.tok.pad_token_id is None:
            # fallback standard pour causal LM
            self.tok.pad_token = self.tok.eos_token

        dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16

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

        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        do_sample = float(temperature) > 0.0

        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
                pad_token_id=self.tok.pad_token_id,
            )

        # ✅ decode only generated part: slice after the full input tensor length
        input_len = enc["input_ids"].shape[1]

        res: List[str] = []
        for i in range(out.shape[0]):
            gen_ids = out[i, input_len:]
            txt = self.tok.decode(gen_ids, skip_special_tokens=True)
            res.append(txt.strip())

        return res