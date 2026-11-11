import dspy

from dspy_weightspace.metrics import extract_answer, is_correct, math_metric, normalize, repetition


def test_normalize_handles_overflow_and_inf():
    assert normalize("1e999") == "1e999"
    assert normalize("infinity") == "infinity"
    assert normalize("1,234.0") == "1234"
    assert normalize("$42") == "42"


def test_is_correct_never_raises_on_weird_answers():
    assert is_correct("1e999", "42") is False
    assert is_correct("nan", "42") is False


def test_extract_prefers_answer_after_think():
    text = "<think>maybe \\boxed{7}</think> So \\boxed{42}."
    assert extract_answer(text) == "42"


def test_repetition_detects_loops():
    ratio, phrase = repetition("Wait, let me re-read the question again. " * 40)
    assert ratio < 0.2 and "re-read" in phrase
    ratio2, _ = repetition(" ".join(str(i) for i in range(700)))
    assert ratio2 > 0.99


def test_metric_scores_and_never_raises():
    gold = dspy.Example(answer="42")
    good = dspy.Prediction(text="<think>ok</think> \\boxed{42}", finish_reason="stop")
    assert math_metric(gold, good).score == 1.0
    weird = dspy.Prediction(text="\\boxed{1e999}", finish_reason="length")
    m = math_metric(gold, weird)
    assert not m.correct and m.score < 1.0
    broken = dspy.Prediction(text=None, finish_reason=None)
    assert math_metric(gold, broken).score <= 1.0


def test_unfinished_thinking_gets_no_credit():
    gold = dspy.Example(answer="42")
    looped = dspy.Prediction(text="so \\boxed{42}. Wait, let me re-check 42", finish_reason="length", thinking=True)
    m = math_metric(gold, looped)
    assert not m.correct and "does not count" in m.feedback
    assert math_metric(gold, dspy.Prediction(text="\\boxed{42}", finish_reason="stop", thinking=False)).correct


def test_group_scores_split_by_what_the_base_model_did():
    from dspy_weightspace.baseline import group_scores

    def row(idx, correct, finish="stop", tokens=10):
        return {"idx": idx, "correct": correct, "finish": finish, "tokens": tokens}

    base = [row(0, True), row(0, False, "length"), row(1, False, "length"), row(1, False, "length")]
    edit = [row(0, True, tokens=4), row(0, True, tokens=6), row(1, True, tokens=20), row(1, False, "length", 30)]
    g = group_scores(base, edit)
    assert g["base_solved"]["problems"] == 1 and g["base_solved"]["accuracy"] == 1.0
    assert g["base_solved"]["tokens_when_correct"] == 5
    assert g["base_looped"]["accuracy"] == 0.5 and g["base_looped"]["cut_rate"] == 0.5
