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
        "mean_tokens": mean(p.n_tokens for p in preds),
        "score": mean(float(m.score) for m in ms),
        "n": len(examples),
    }


def results_table(rows: dict[str, dict]) -> str:
    head = "| Variant | Accuracy | Cut at limit | Mean tokens | Score | n |\n|---|---|---|---|---|---|"
    body = "\n".join(
        f"| {k} | {v['accuracy']:.1%} | {v['cut_rate']:.1%} | {v['mean_tokens']:.0f} | {v['score']:.3f} | {v['n']} |"
        for k, v in rows.items()
    )
    return head + "\n" + body
