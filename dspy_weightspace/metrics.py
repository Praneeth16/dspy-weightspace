"""GEPA-shaped metric: (gold, pred, trace, pred_name, pred_trace) -> Prediction(score, feedback).

Three signals per answer: correct, terminated, not looping. Accuracy is what counts (0.96); termination and
non-looping are 0.02 tie-breakers so equal-accuracy candidates prefer the one that stops repeating. Earlier weights
of 0.4 / 0.3 / 0.3 let the optimizer raise the score while accuracy fell (see evidence/air_run4_warmstart).
The feedback text still names loops and cut-offs explicitly; that is where the reflection model reads them.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

import dspy


def boxed_all(text: str) -> list[str]:
    """Contents of every \\boxed{...}, braces matched to any depth (MATH answers nest: \\frac{\\sqrt{3}}{2})."""
    out, start = [], 0
    while (i := text.find("\\boxed{", start)) != -1:
        j, depth = i + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth:
            break
        out.append(text[i + len("\\boxed{") : j - 1])
        start = j
    return out


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
LOOP_NGRAM, LOOP_TAIL_WORDS, LOOP_THRESHOLD = 8, 600, 0.6
W_CORRECT, W_TERMINATED, W_NOT_LOOPING = 0.96, 0.02, 0.02


def extract_answer(text: str) -> str | None:
    final = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    boxed = boxed_all(final) or boxed_all(text)
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
    except ValueError:
        return s
    if not math.isfinite(f):
        return s
    return str(int(f)) if f == int(f) else str(f)


def is_correct(pred: str | None, gold: str) -> bool:
    """Normalized string match, then math_verify when installed (the final eval's grader: \\frac{1}{2} == 0.5)."""
    if pred is None:
        return False
    if normalize(pred) == normalize(gold):
        return True
    try:
        from math_verify import parse, verify
    except ImportError:
        return False
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    # Timeouts off: math_verify's own SIGALRM would clash with the sandbox's, and answers here are short strings.
    g, t = parse(f"${gold}$", parsing_timeout=None), parse(f"${pred}$", parsing_timeout=None)
    return bool(g and t and verify(g, t, timeout_seconds=None))


def repetition(text: str, k: int = LOOP_NGRAM, tail: int = LOOP_TAIL_WORDS) -> tuple[float, str]:
    """(distinct k-gram ratio over the last `tail` words, most repeated k-gram). Ratio 1.0 = no repetition."""
    words = text.split()[-tail:]
    if len(words) <= k:
        return 1.0, ""
    grams = Counter(tuple(words[i : i + k]) for i in range(len(words) - k))
    top, count = grams.most_common(1)[0]
    return len(grams) / max(1, sum(grams.values())), (" ".join(top) if count > 2 else "")


def math_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    """Never raises: a metric exception inside GEPA's valset pass would discard the candidate being scored."""
    try:
        return _math_metric(gold, pred)
    except Exception as e:  # noqa: BLE001
        return dspy.Prediction(
            score=0.0,
            feedback=f"metric error: {type(e).__name__}: {e}",
            correct=False,
            terminated=False,
            looping=False,
            repetition_ratio=1.0,
        )


def _math_metric(gold, pred) -> dspy.Prediction:
    text = getattr(pred, "text", "") or ""
    finish = getattr(pred, "finish_reason", "stop")
    provisional = extract_answer(text)
    # As in the post's evaluate.py: with thinking on, an answer counts only once thinking has closed. Otherwise a
    # trace cut mid-loop scores whenever a \boxed{} (or the right number) appeared somewhere in the loop.
    answer = provisional if (not getattr(pred, "thinking", True) or "</think>" in text) else None
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
    if answer is None and provisional is not None:
        match = "matches" if is_correct(provisional, gold.answer) else "does not match"
        fb += f" Provisional answer seen in unfinished thinking: {provisional!r} ({match} gold); it does not count."
    return dspy.Prediction(
        score=score, feedback=fb, correct=correct, terminated=terminated, looping=looping, repetition_ratio=ratio
    )
