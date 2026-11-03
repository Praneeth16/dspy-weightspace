"""GEPA-shaped metric: (gold, pred, trace, pred_name, pred_trace) -> Prediction(score, feedback).

Score = 0.5 * correct + 0.5 * terminated. The blog judged edits by accuracy alone at 32k tokens; at notebook
budgets (2k tokens) most failures are cut-offs, so termination is rewarded explicitly and reported separately.
"""

from __future__ import annotations

import re

import dspy

BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    boxed = BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    tail = text.rsplit("</think>", 1)[-1]
    nums = NUMBER.findall(tail)
    return nums[-1] if nums else None


def normalize(s: str) -> str:
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def is_correct(pred: str | None, gold: str) -> bool:
    return pred is not None and normalize(pred) == normalize(gold)


def math_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    text = getattr(pred, "text", "") or ""
    finish = getattr(pred, "finish_reason", "stop")
    answer = extract_answer(text)
    correct = is_correct(answer, gold.answer)
    terminated = finish != "length"
    score = 0.5 * float(correct) + 0.5 * float(terminated)
    if not terminated:
        tail = text[-300:].replace("\n", " ")
        fb = f"Cut off at the token limit without a final answer. Last 300 chars: ...{tail}"
    elif correct:
        fb = "Correct and terminated."
    else:
        fb = f"Terminated but wrong: extracted {answer!r}, expected {gold.answer!r}."
    return dspy.Prediction(score=score, feedback=fb, correct=correct, terminated=terminated)
