"""The DSPy program under optimization. Its weights, not its prompt, are what the optimizer changes."""

from __future__ import annotations

import dspy

from .data import to_chat
from .hf_lm import HFLocalLM


class MathSolver(dspy.Module):
    def __init__(self, lm: HFLocalLM, max_new_tokens: int | None = None):
        super().__init__()
        self.lm = lm
        self.max_new_tokens = max_new_tokens

    def forward(self, problem: str) -> dspy.Prediction:
        return self.batch_forward([problem])[0]

    def batch_forward(
        self,
        problems: list[str],
        seed: int | None = 0,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[dspy.Prediction]:
        comps = self.lm.generate_detailed(
            [to_chat(p) for p in problems],
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
        return [dspy.Prediction(text=c.text, finish_reason=c.finish_reason, n_tokens=c.n_tokens) for c in comps]
