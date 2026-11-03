"""GEPA-shaped metric: (gold, pred, trace, pred_name, pred_trace) -> Prediction(score, feedback).

Three signals per answer: correct, terminated, not looping. The blog judged edits by accuracy at 32k tokens. At
notebook budgets (2k-4k tokens) almost nothing terminates, so the loop detector carries the signal the blog's
"cut at limit" carried: a low distinct n-gram ratio over the tail of the generation.
"""

from __future__ import annotations

import re
from collections import Counter

import dspy

BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
LOOP_NGRAM, LOOP_TAIL_WORDS, LOOP_THRESHOLD = 8, 600, 0.6
W_CORRECT, W_TERMINATED, W_NOT_LOOPING = 0.4, 0.3, 0.3


def extract_answer(text: str) -> str | None:
    final = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    boxed = BOXED.findall(final) or BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = NUMBER.findall(final)
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


def repetition(text: str, k: int = LOOP_NGRAM, tail: int = LOOP_TAIL_WORDS) -> tuple[float, str]:
    """(distinct k-gram ratio over the last `tail` words, most repeated k-gram). Ratio 1.0 = no repetition."""
    words = text.split()[-tail:]
    if len(words) <= k:
        return 1.0, ""
    grams = Counter(tuple(words[i : i + k]) for i in range(len(words) - k))
    top, count = grams.most_common(1)[0]
    return len(grams) / max(1, sum(grams.values())), (" ".join(top) if count > 2 else "")


def math_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    text = getattr(pred, "text", "") or ""
    finish = getattr(pred, "finish_reason", "stop")
    answer = extract_answer(text)
    correct = is_correct(answer, gold.answer)
    terminated = finish != "length"
    ratio, phrase = repetition(text)
    looping = ratio < LOOP_THRESHOLD
    score = W_CORRECT * correct + W_TERMINATED * terminated + W_NOT_LOOPING * (not looping)

    if looping:
        fb = f"Looping: distinct 8-gram ratio {ratio:.2f} over the tail; repeated phrase: {phrase[:120]!r}."
    elif not terminated:
        fb = f"Cut off at the token limit without finishing (no loop detected, ratio {ratio:.2f}). "
        fb += f"Last 200 chars: ...{text[-200:].replace(chr(10), ' ')!r}"
    elif correct:
        fb = "Correct and terminated."
    else:
        fb = f"Terminated but wrong: extracted {answer!r}, expected {gold.answer!r}."
    if not terminated and answer is not None:
        match = "matches" if correct else "does not match"
        fb += f" Provisional boxed answer seen in thinking: {answer!r} ({match} gold)."
    return dspy.Prediction(
        score=score, feedback=fb, correct=correct, terminated=terminated, looping=looping, repetition_ratio=ratio
    )
