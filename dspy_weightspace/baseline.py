"""Decoding baselines and the results table. Answers the critics' question in the same table as the edits."""

from __future__ import annotations

from statistics import mean

import dspy

from .program import MathSolver

BLOG_T, CARD_T = 0.6, 1.0


def score_rows(program: MathSolver, examples: list[dspy.Example], metric, **gen) -> dict:
    preds = program.batch_forward([e.problem for e in examples], seed=0, **gen)
    ms = [metric(e, p) for e, p in zip(examples, preds, strict=True)]
    return {
        "accuracy": mean(float(m.correct) for m in ms),
        "cut_rate": mean(float(not m.terminated) for m in ms),
        "loop_rate": mean(float(m.looping) for m in ms),
        "mean_tokens": mean(p.n_tokens for p in preds),
        "score": mean(float(m.score) for m in ms),
        "n": len(examples),
    }


def results_table(rows: dict[str, dict]) -> str:
    head = "| Variant | Accuracy | Looping | Cut at limit | Mean tokens | Score | n |\n|---|---|---|---|---|---|---|"
    body = "\n".join(
        f"| {k} | {v['accuracy']:.1%} | {v['loop_rate']:.1%} | {v['cut_rate']:.1%} | {v['mean_tokens']:.0f} "
        f"| {v['score']:.3f} | {v['n']} |"
        for k, v in rows.items()
    )
    return head + "\n" + body


def group_scores(base_rows: list[dict], rows: list[dict]) -> dict[str, dict]:
    """Score a result file separately on the problems the base model solved at least once and on the problems where
    every base sample hit the token limit. A wrap-up edit should fix the second group without hurting the first.

    Rows are vllm_eval records (idx, correct, finish, tokens), several samples per problem."""
    by: dict[int, list[dict]] = {}
    for r in base_rows:
        by.setdefault(r["idx"], []).append(r)
    groups = {
        "base_solved": {i for i, v in by.items() if any(r["correct"] for r in v)},
        "base_looped": {i for i, v in by.items() if all(r["finish"] == "length" for r in v)},
    }
    out = {}
    for name, ids in groups.items():
        sub = [r for r in rows if r["idx"] in ids]
        right = [r for r in sub if r["correct"]]
        out[name] = {
            "problems": len(ids),
            "accuracy": mean(float(r["correct"]) for r in sub) if sub else float("nan"),
            "cut_rate": mean(float(r["finish"] == "length") for r in sub) if sub else float("nan"),
            "mean_tokens": mean(r["tokens"] for r in sub) if sub else float("nan"),
            "tokens_when_correct": mean(r["tokens"] for r in right) if right else float("nan"),
        }
    return out
