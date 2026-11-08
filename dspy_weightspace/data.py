"""GSM8K practice set: small, fixed, seeded. None of it is MATH-500."""

from __future__ import annotations

import random
import re

import dspy

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_gsm8k(n_dev: int = 48, n_holdout: int = 48, seed: int = 0) -> tuple[list[dspy.Example], list[dspy.Example]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    rows = [ds[i] for i in idx[: n_dev + n_holdout]]
    exs = [
        dspy.Example(
            problem=r["question"],
            answer=r["answer"].rsplit("####", 1)[-1].strip(),
            rationale=re.sub(r"<<[^>]*>>", "", r["answer"].rsplit("####", 1)[0]).strip(),
        ).with_inputs("problem")
        for r in rows
    ]
    return exs[:n_dev], exs[n_dev:]


def to_chat(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": problem + SUFFIX}]
