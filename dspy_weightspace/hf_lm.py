"""In-process Hugging Face language model for DSPy with a patchable weight overlay.

DSPy 3.4.1 ships no in-process transformers LM. Its ``LocalProvider`` runs an SGLang server and
restarts it per checkpoint, which is far too slow when an optimizer tries dozens of weight patches.
``HFLocalLM`` keeps the model in memory, keeps a CPU copy of the original tensors, and lets the
optimizer swap edited tensors in and out between evaluations.

Implements the legacy ``BaseLM.forward`` contract (supported through DSPy 3.4, scheduled for removal
in 3.5). Porting to the ``Engine.complete(Request)`` protocol is a follow-up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from dspy.clients.base_lm import BaseLM

FORBIDDEN_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")
FINAL_NORM_SUFFIX = ".norm.weight"


@dataclass
class Completion:
    text: str
    finish_reason: str
    n_tokens: int


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def is_editable(name: str) -> bool:
    """Blog's scope rule: no embeddings, no output head, no final norm."""
    if name.endswith(FORBIDDEN_SUFFIXES):
        return False
    if name.endswith(FINAL_NORM_SUFFIX) and ".layers." not in name:
        return False
    return True


class HFLocalLM(BaseLM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        revision: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        enable_thinking: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        batch_size: int = 16,
        **kwargs: Any,
    ):
        super().__init__(model=f"hf-local/{model_name}", temperature=temperature, max_tokens=max_tokens, cache=False)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or _pick_device()
        self.dtype = dtype or _pick_dtype(self.device)
        self.enable_thinking = enable_thinking
        self.top_p, self.top_k, self.repetition_penalty, self.batch_size = top_p, top_k, repetition_penalty, batch_size

        self.tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.hf = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=self.dtype)
        self.hf.to(self.device).eval()
        for p in self.hf.parameters():
            p.requires_grad_(False)

        self._params = dict(self.hf.named_parameters())
        self.names = sorted(self._params)
        self._base = {n: p.detach().to("cpu", copy=True) for n, p in self._params.items()}
        self._edited: set[str] = set()
        self.passes = 0

    # ----- weights ----------------------------------------------------------------------------------
    def resolve(self, name: str) -> str:
        """Accept wrapper names (model.language_model.layers.N...) and text-model names (model.layers.N...)."""
        if name in self._params:
            return name
        alt = (
            name.replace("model.language_model.", "model.", 1)
            if "language_model." in name
            else name.replace("model.", "model.language_model.", 1)
        )
        if alt in self._params:
            return alt
        raise KeyError(name)

    def original(self, name: str) -> torch.Tensor:
        return self._base[self.resolve(name)].detach().clone().float()

    def current(self, name: str) -> torch.Tensor:
        return self._params[self.resolve(name)].detach().to("cpu", copy=True).float()

    def apply_patch(self, edits: dict[str, torch.Tensor]) -> None:
        for name, t in edits.items():
            name = self.resolve(name)
            if not is_editable(name):
                raise ValueError(f"{name} is out of scope")
            p = self._params[name]
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{name}: shape {tuple(t.shape)} != {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(t.to(device=p.device, dtype=p.dtype))
            self._edited.add(name)

    def reset(self) -> None:
        with torch.no_grad():
            for name in self._edited:
                self._params[name].copy_(self._base[name].to(self._params[name].device))
        self._edited.clear()

    def edited(self) -> list[str]:
        return sorted(self._edited)

    def changed_entries(self) -> int:
        n = 0
        for name in self._edited:
            cur = self._params[name].detach().to("cpu")
            n += int((cur != self._base[name]).sum())
        return n

    # ----- generation -------------------------------------------------------------------------------
    def _render(self, messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
        )

    @torch.no_grad()
    def generate_detailed(
        self,
        chats: list[list[dict[str, str]]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        **gen_kwargs: Any,
    ) -> list[Completion]:
        """Batched generation that keeps finish reasons. One call counts as one forward pass."""
        max_new_tokens = max_new_tokens or self.kwargs["max_tokens"]
        temperature = self.kwargs["temperature"] if temperature is None else temperature
        prompts = [self._render(c) for c in chats]
        out: list[Completion] = []
        self.passes += 1
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tok(prompts[i : i + self.batch_size], return_tensors="pt", padding=True).to(self.device)
            if seed is not None:
                torch.manual_seed(seed + i)
            gen = self.hf.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=self.top_p if temperature > 0 else None,
                top_k=self.top_k if temperature > 0 else None,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tok.pad_token_id,
                **gen_kwargs,
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            for row in new:
                ids = row.tolist()
                eos = {self.tok.eos_token_id, self.tok.pad_token_id}
                n = next((k for k, t in enumerate(ids) if t in eos), len(ids))
                finish = "stop" if n < len(ids) or n < max_new_tokens else "length"
                out.append(Completion(self.tok.decode(ids[:n], skip_special_tokens=True), finish, n))
        return out

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        n = int(kwargs.get("n", 1))
        comps = self.generate_detailed(
            [messages] * n, max_new_tokens=kwargs.get("max_tokens"), temperature=kwargs.get("temperature")
        )
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=c.text, tool_calls=None, reasoning_content=None),
                finish_reason=c.finish_reason,
                index=i,
            )
            for i, c in enumerate(comps)
        ]
        return SimpleNamespace(
            choices=choices,
            model=self.model,
            usage={"completion_tokens": sum(c.n_tokens for c in comps)},
            created=int(time.time()),
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    # ----- activations ------------------------------------------------------------------------------
    def _layers(self):
        m = self.hf
        for attr in ("model.language_model.layers", "model.layers"):
            obj = m
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError("could not locate decoder layers")

    @torch.no_grad()
    def hidden_states(
        self,
        chats: list[list[dict[str, str]]],
        layers: list[int],
        completions: list[str] | None = None,
        max_completion_tokens: int = 1024,
        chunk: int = 4,
        last_tokens: int | None = None,
    ):
        """Teacher-forced residual stream after each requested layer, mean over the completion tokens.

        Completions are truncated to `max_completion_tokens` and texts processed `chunk` at a time: eager attention
        on full-attention layers is quadratic in length, and a 24 x 4k batch exhausts a 24 GB GPU.
        `last_tokens` averages only the final N completion tokens (where a loop, if any, has taken over).
        Returns {layer: tensor[n, hidden]} on CPU float32. One call counts as one forward pass.
        """
        self.passes += 1
        texts = []
        for i, c in enumerate(chats):
            comp = completions[i] if completions else ""
            ids = self.tok(comp, add_special_tokens=False)["input_ids"][:max_completion_tokens]
            texts.append(self._render(c) + self.tok.decode(ids))
        prompt_lens = [len(self.tok(self._render(c))["input_ids"]) for c in chats]
        res: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        for start in range(0, len(texts), chunk):
            enc = self.tok(texts[start : start + chunk], return_tensors="pt", padding=True).to(self.device)
            out = self.hf(**enc, output_hidden_states=True)
            mask = enc["attention_mask"]
            total = mask.shape[1]
            for layer in layers:
                h = out.hidden_states[layer].float()
                for i in range(h.shape[0]):
                    begin = total - int(mask[i].sum()) + prompt_lens[start + i]
                    if last_tokens:
                        begin = max(begin, total - last_tokens)
                    seg = h[i, begin:] if begin < total else h[i, -1:]
                    res[layer].append(seg.mean(0).cpu())
            del out, enc
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return {layer: torch.stack(rows) for layer, rows in res.items()}
