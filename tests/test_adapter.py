"""End-to-end optimizer loop with a fake model and a scripted reflection LM."""

from __future__ import annotations

import dspy

from dspy_weightspace.adapter import COMPONENT, WeightSpaceAdapter
from dspy_weightspace.metrics import math_metric
from dspy_weightspace.optimizer import WeightSpaceOptimizer
from dspy_weightspace.probes import Probes
from dspy_weightspace.program import MathSolver

from .conftest import FakeLM

L1 = "model.language_model.layers.1.mlp.down_proj.weight"
PATCH = f"w = load_original({L1!r}); w[:, :2] = 0; save_edit({L1!r}, w)"


def examples(n=6):
    return [dspy.Example(problem=f"What is {i} + {42 - i}?", answer="42").with_inputs("problem") for i in range(n)]


class ScriptedLM:
    """Replies in order: one probe, one code block, then the patch."""

    def __init__(self):
        self.replies = [
            '```forward\n{"type": "generate", "model": "original", "problems": ["p000", "p001"]}\n```',
            "```python\nprint(len(NAMES))\n```",
            f"```patch\n{PATCH}\n```\nRemoved two columns.",
        ]
        self.calls = []

    def __call__(self, messages=None, **kw):
        self.calls.append(messages)
        return [self.replies.pop(0) if self.replies else "```patch\n" + PATCH + "\n```"]


def test_adapter_scores_and_reflects(tmp_path):
    lm = FakeLM()
    prog = MathSolver(lm, max_new_tokens=64)
    adapter = WeightSpaceAdapter(lm, prog, math_metric, str(tmp_path), proposer=None)
    batch = examples(4)
    base = adapter.evaluate(batch, {COMPONENT: ""}, capture_traces=True)
    assert base.scores == [0.0] * 4
    refl = adapter.make_reflective_dataset({COMPONENT: ""}, base, [COMPONENT])
    assert "Looping" in refl[COMPONENT][0]["Feedback"]
    fixed = adapter.evaluate(batch, {COMPONENT: PATCH}, capture_traces=False)
    assert fixed.scores == [1.0] * 4
    assert lm.changed_entries() == 0  # reset after evaluate


def test_broken_patch_gets_failure_score(tmp_path):
    lm = FakeLM()
    adapter = WeightSpaceAdapter(lm, MathSolver(lm), math_metric, str(tmp_path), proposer=None)
    out = adapter.evaluate(examples(2), {COMPONENT: "raise RuntimeError('boom')"}, capture_traces=True)
    assert out.scores == [0.0, 0.0] and "boom" in out.trajectories[0]["feedback"]


def test_optimizer_finds_patch(tmp_path):
    lm = FakeLM()
    prog = MathSolver(lm, max_new_tokens=64)
    scripted = ScriptedLM()
    opt = WeightSpaceOptimizer(
        math_metric,
        scripted,
        max_metric_calls=40,
        passes_per_round=2,
        reflection_minibatch_size=2,
        log_dir=str(tmp_path / "run"),
        max_turns=6,
    )
    train, val = examples(4), examples(4)
    out = opt.compile(prog, trainset=train, valset=val)
    assert PATCH in out.patch_src
    assert out.weights_changed == 8 * 2
    assert out.edited_tensors == [L1]
    assert (tmp_path / "run" / "round_01" / "transcript.md").exists()
    assert len(scripted.calls) >= 3


def test_probes_budget(tmp_path):
    lm = FakeLM()
    prog = MathSolver(lm, max_new_tokens=64)
    probes = Probes(lm, prog, math_metric, examples(3), str(tmp_path), budget=1, max_items=4)
    s1 = probes.dispatch({"type": "generate", "model": "original", "problems": ["p000"]})
    assert "pass 1 (0 left)" in s1 and "cut_at_limit=1.000" in s1
    s2 = probes.dispatch({"type": "generate", "model": "original", "problems": ["p000"]})
    assert s2.startswith("error: forward-pass budget exhausted")


def test_python_and_forward_in_one_reply_both_run(tmp_path):
    """An edit block followed by a probe in the same reply must apply the edit before the probe runs."""
    lm = FakeLM()
    prog = MathSolver(lm, max_new_tokens=64)
    probes = Probes(lm, prog, math_metric, examples(3), str(tmp_path / "p"), budget=2, max_items=4)
    from dspy_weightspace.proposer import WeightPatchProposer

    class OneShot:
        def __init__(self):
            self.replies = [
                f"```python\n{PATCH}\n```\n"
                '```forward\n{"type": "generate", "model": "both", "problems": ["p000"]}\n```',
                f"```patch\n{PATCH}\n```",
            ]

        def __call__(self, messages=None, **kw):
            return [self.replies.pop(0)]

    prop = WeightPatchProposer(OneShot(), lm, probes, str(tmp_path), max_turns=4)
    out = prop({COMPONENT: ""}, {COMPONENT: []}, [COMPONENT])
    assert PATCH in out[COMPONENT]
    t = (tmp_path / "round_01" / "transcript.md").read_text()
    assert "[edited: ['" in t and "pass 1 (1 left)" in t and "edited: n=1" in t


def test_compile_without_valset_holds_out_selection_examples(tmp_path):
    lm = FakeLM()
    opt = WeightSpaceOptimizer(
        math_metric,
        ScriptedLM(),
        max_metric_calls=20,
        passes_per_round=2,
        reflection_minibatch_size=2,
        log_dir=str(tmp_path / "run"),
        max_turns=6,
    )
    out = opt.compile(MathSolver(lm, max_new_tokens=64), trainset=examples(6))
    assert len(out.detailed_results.val_subscores[0]) == 4


def test_patch_that_cannot_be_applied_uses_budget(tmp_path):
    """A backend that refuses a patch once made GEPA retry the same evaluation until the job timed out."""

    class Refusing:
        def set_weights(self, tensors):
            raise KeyError("layers.1.mlp.gate_proj.weight: no matching vLLM parameter")

    lm = FakeLM()
    lm.gen_backend = Refusing()
    adapter = WeightSpaceAdapter(lm, MathSolver(lm, max_new_tokens=64), math_metric, str(tmp_path), proposer=None)
    out = adapter.evaluate(examples(3), {COMPONENT: PATCH}, capture_traces=True)
    assert out.scores == [0.0] * 3 and out.num_metric_calls == 3
    assert "could not be applied" in out.trajectories[0]["feedback"]
    assert lm.edited() == []
