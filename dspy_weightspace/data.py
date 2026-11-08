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


def load_math(
    n_dev: int = 48, n_holdout: int = 48, seed: int = 0, levels: tuple[int, ...] = (3, 4, 5)
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """MATH train split (MATH-500 is drawn from the test split, so none of these are test problems), harder levels
    only: at levels 1-2 the base model rarely loops, so they say little about a loop edit."""
    from datasets import load_dataset

    from .metrics import boxed_all

    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
    keep = {f"Level {k}" for k in levels}
    idx = [i for i, lv in enumerate(ds["level"]) if lv in keep]
    random.Random(seed).shuffle(idx)
    exs = []
    for i in idx:
        r = ds[i]
        boxed = boxed_all(r["solution"])
        if boxed:
            exs.append(
                dspy.Example(problem=r["problem"], answer=boxed[-1].strip(), rationale=r["solution"]).with_inputs(
                    "problem"
                )
            )
        if len(exs) == n_dev + n_holdout:
            break
    return exs[:n_dev], exs[n_dev:]
