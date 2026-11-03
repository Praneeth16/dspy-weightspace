"""A fake HFLocalLM: real tensors, no transformers. Enough to exercise the sandbox, adapter and GEPA loop."""

from __future__ import annotations

import torch

from dspy_weightspace.hf_lm import Completion, HFLocalLM


class FakeTok:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, **kw):
        return messages[-1]["content"]


class FakeLM(HFLocalLM):
    def __init__(self, loops: bool = True):
        self.model = "hf-local/fake"
        self.model_type = "chat"
        self.cache = False
        self.callbacks = []
        self.num_retries = 0
        self.kwargs = {"temperature": 0.6, "max_tokens": 64}
        self.history = []
        self.model_name = "fake"
        self.device, self.dtype = "cpu", torch.float32
        self.enable_thinking = True
        self.top_p, self.top_k, self.repetition_penalty, self.batch_size = 0.95, 20, 1.0, 8
        self.tok = FakeTok()
        g = torch.Generator().manual_seed(0)
        self._params = {
            "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, generator=g),
            "model.language_model.layers.1.mlp.down_proj.weight": torch.randn(8, 16, generator=g),
            "model.language_model.norm.weight": torch.ones(8),
            "model.language_model.embed_tokens.weight": torch.randn(32, 8, generator=g),
        }
        self.names = sorted(self._params)
        self._base = {n: p.clone() for n, p in self._params.items()}
        self._edited = set()
        self.passes = 0
        self.loops = loops

    def generate_detailed(self, chats, max_new_tokens=None, temperature=None, seed=None, **kw):
        self.passes += 1
        # "Fixed" when layer 1 down_proj has been edited: the fake stops looping.
        fixed = "model.language_model.layers.1.mlp.down_proj.weight" in self._edited
        out = []
        for _ in chats:
            if self.loops and not fixed:
                out.append(Completion("<think>" + "Wait, let me re-read the question again. " * 40, "length", 64))
            else:
                out.append(Completion("<think>ok</think> The answer is \\boxed{42}.", "stop", 12))
        return out

    def hidden_states(self, chats, layers, completions=None):
        self.passes += 1
        return {layer: torch.zeros(len(chats), 8) for layer in layers}

    def _layers(self):
        return [None, None]
