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

        # Decoder-only models require left padding for correct batch generation
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
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

        max_len = getattr(self.tok, "model_max_length", None)
        if max_len is None or max_len > 32768:
            max_len = 4096

        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
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
    # ------------------------------------------------------------------
    # Ajouts : classification par probabilité des réponses (pas de parsing)
    # ------------------------------------------------------------------

    def build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Prompt complet (template de chat du modèle), prêt pour la réponse de l'assistant."""
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score_options(self, prompts: List[str], options: List[str], batch_size: int = 4) -> List[List[float]]:
        """Pour chaque prompt, log-probabilité que l'assistant réponde exactement chacune des
        `options` (somme des log-probas de ses tokens). Une seule passe avant par
        (prompt, option), sans génération : aucune réponse mal formée possible, et les
        probabilités servent de score de confiance.
        Retour : liste (par prompt) de listes (par option) de log-probabilités."""
        torch = self.torch
        opt_ids = [self.tok(o, add_special_tokens=False)["input_ids"] for o in options]
        K = max(len(x) for x in opt_ids)
        pad_id = self.tok.pad_token_id
        results: List[List[float]] = []

        for start in range(0, len(prompts), max(1, int(batch_size))):
            batch = prompts[start:start + max(1, int(batch_size))]
            seqs, owners = [], []
            for bi, p in enumerate(batch):
                p_ids = self.tok(p, add_special_tokens=False)["input_ids"]
                for oi, o in enumerate(opt_ids):
                    seqs.append(p_ids + o)
                    owners.append((bi, oi))
            L = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, s in enumerate(seqs):          # padding à gauche : les options finissent toutes en position L-1
                input_ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
                attn[i, L - len(s):] = 1
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            dev = self.model.device
            input_ids, attn, pos = input_ids.to(dev), attn.to(dev), pos.to(dev)

            with torch.inference_mode():
                try:   # ne calculer les logits que sur les K+1 dernières positions (économie mémoire)
                    out = self.model(input_ids=input_ids, attention_mask=attn, position_ids=pos, logits_to_keep=K + 1)
                    logits = out.logits[:, -(K + 1):, :]
                except TypeError:
                    out = self.model(input_ids=input_ids, attention_mask=attn, position_ids=pos)
                    logits = out.logits[:, -(K + 1):, :]
                logp = torch.log_softmax(logits.float(), dim=-1)

            scores = [[0.0] * len(options) for _ in batch]
            for row, (bi, oi) in enumerate(owners):
                o = opt_ids[oi]
                n = len(o)
                # le token o[t] (position absolue L-n+t) est prédit par la position L-n+t-1,
                # soit l'indice K-n+t dans la fenêtre des K+1 dernières positions
                scores[bi][oi] = float(sum(logp[row, K - n + t, o[t]].item() for t in range(n)))
            results.extend(scores)
            del out, logits, logp
        return results

    def option_first_tokens(self, options: List[str]) -> List[dict]:
        """Tokenisation de chaque option, telle qu'elle suivrait le prompt (contrôle : une
        option doit idéalement tenir en un seul token)."""
        out = []
        for o in options:
            ids = self.tok(o, add_special_tokens=False)["input_ids"]
            out.append({"option": o, "ids": ids, "tokens": self.tok.convert_ids_to_tokens(ids)})
        return out

    def next_token_logprobs(self, prompts: List[str], options: List[str], batch_size: int = 8) -> List[List[float]]:
        """Log-probabilité du PREMIER token de chaque option comme prochain token après le
        prompt. Une seule passe avant par prompt (au lieu d'une par option) ; à utiliser avec
        des options d'un seul token (ex. "A", "B", "C"), cf. option_first_tokens()."""
        torch = self.torch
        first = [self.tok(o, add_special_tokens=False)["input_ids"][0] for o in options]
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        results: List[List[float]] = []
        bs = max(1, int(batch_size))
        for start in range(0, len(prompts), bs):
            batch = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts[start:start + bs]]
            L = max(len(s) for s in batch)
            input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), L), dtype=torch.long)
            for i, s in enumerate(batch):          # padding à gauche : dernier token en L-1
                input_ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
                attn[i, L - len(s):] = 1
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            dev = self.model.device
            input_ids, attn, pos = input_ids.to(dev), attn.to(dev), pos.to(dev)
            with torch.inference_mode():
                try:
                    out = self.model(input_ids=input_ids, attention_mask=attn, position_ids=pos, logits_to_keep=1)
                except TypeError:
                    out = self.model(input_ids=input_ids, attention_mask=attn, position_ids=pos)
                logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            for i in range(len(batch)):
                results.append([float(logp[i, t].item()) for t in first])
            del out, logp
        return results

    def generate_from_prompts(self, prompts: List[str], max_new_tokens: int = 80, batch_size: int = 4) -> List[str]:
        """Génération gloutonne à partir de prompts déjà formatés (ex. prompt + début de réponse)."""
        res: List[str] = []
        for start in range(0, len(prompts), max(1, int(batch_size))):
            batch = prompts[start:start + max(1, int(batch_size))]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            # seulement input_ids / attention_mask (certains tokenizers ajoutent token_type_ids,
            # que generate() refuse)
            enc = {k: enc[k].to(self.model.device) for k in ("input_ids", "attention_mask")}
            with self.torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=int(max_new_tokens), do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            n_in = enc["input_ids"].shape[1]
            for i in range(out.shape[0]):
                res.append(self.tok.decode(out[i, n_in:], skip_special_tokens=True).strip())
        return res
