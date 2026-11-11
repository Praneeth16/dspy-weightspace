from __future__ import annotations

import json

import dspy
import pytest
import torch

from dspy_weightspace.adapter import COMPONENT
from dspy_weightspace.metrics import math_metric
from dspy_weightspace.optimizer import WeightSpaceOptimizer
from dspy_weightspace.probes import Probes
from dspy_weightspace.program import MathSolver
from dspy_weightspace.proposer import WeightPatchProposer
from dspy_weightspace.wrapup import WrapUpProbe, boundary_mask

from .conftest import FakeLM
from .test_adapter import PATCH, ScriptedLM, examples

transformers = pytest.importorskip("transformers")


class CharTok:
    """One token per character, so "S" (as in "So") and "W" (as in "Wait") are single tokens."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 128 for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class TinyLM:
    def __init__(self):
        cfg = transformers.GPT2Config(
            vocab_size=128,
            n_positions=512,
            n_embd=16,
            n_layer=1,
            n_head=2,
            initializer_range=0.2,
            bos_token_id=0,
            eos_token_id=0,
        )
        torch.manual_seed(0)
        self.hf = transformers.GPT2LMHeadModel(cfg).eval()
        self.device, self.tok, self._edited = "cpu", CharTok(), set()

    def _render(self, chat):
        return chat[0]["content"]


TRACES = [("What is 2+2?", "It is four. Wait. It is four. Wait. It is four. So done.")]


def test_boundary_mask_marks_the_token_after_a_sentence_end():
    tok = CharTok()
    assert boundary_mask(tok, tok("a. b")["input_ids"]).tolist() == [False, False, True, False]


def test_probe_is_one_on_base_and_rises_when_a_closer_is_boosted(monkeypatch):
    monkeypatch.setattr("dspy_weightspace.wrapup.CLOSERS", ["S"])
    monkeypatch.setattr("dspy_weightspace.wrapup.STARTERS", ["W"])
    monkeypatch.setattr("dspy_weightspace.wrapup.THINK_CLOSE", "S")
    lm = TinyLM()
    probe = WrapUpProbe(lm, TRACES, max_tokens=64, chunk=16)
    base = probe.measure()
    assert base["closers_x"] == pytest.approx(1.0) and base["kl_productive"] == pytest.approx(0.0, abs=1e-6)
    # Push the last block's output toward the token "S"; the output head itself stays untouched.
    direction = lm.hf.get_output_embeddings().weight[ord("S")].detach()
    with torch.no_grad():
        lm.hf.transformer.h[0].mlp.c_proj.bias.add_(2.0 * direction)
    lm._edited.add("x")
    edited = probe.measure()
    assert edited["closers_x"] > 1.2 and edited["kl_productive"] > 0 and edited["think_close_dlogp"] > 0
    assert "closers_x=" in probe.summary()


def test_probe_refuses_to_take_an_edited_model_as_its_baseline():
    lm = TinyLM()
    lm._edited.add("x")
    with pytest.raises(ValueError):
        WrapUpProbe(lm, TRACES)


class StubWrapUp:
    def summary(self):
        return "closers_x=4.00 starters_x=0.80"


def test_wrap_up_request_is_free_capped_and_off_by_default(tmp_path):
    lm = FakeLM()
    prog = MathSolver(lm, max_new_tokens=64)
    off = Probes(lm, prog, math_metric, examples(3), str(tmp_path / "a"), budget=1)
    assert off.dispatch({"type": "wrap_up"}) == "error: type must be 'generate' or 'teacher_forced'"
    on = Probes(lm, prog, math_metric, examples(3), str(tmp_path / "b"), budget=1, wrapup=StubWrapUp(), wrapup_cap=2)
    out = on.dispatch({"type": "wrap_up", "model": "original"})
    assert "closers_x=4.00" in out and "(1 left)" in out and on.left == 1  # the forward-pass budget is untouched
    on.dispatch({"type": "wrap_up", "model": "original"})
    assert on.dispatch({"type": "wrap_up", "model": "original"}).startswith("error: wrap_up limit")
    assert "no edits saved yet" in on.dispatch({"type": "wrap_up", "model": "edited"}) or on.wrapup_used == 2


def test_prompt_mentions_the_tool_only_when_the_probe_is_on(tmp_path):
    lm = FakeLM()
    prog = MathSolver(lm, max_new_tokens=64)
    off = WeightPatchProposer(None, lm, Probes(lm, prog, math_metric, examples(3), str(tmp_path / "a")), str(tmp_path))
    on = WeightPatchProposer(
        None, lm, Probes(lm, prog, math_metric, examples(3), str(tmp_path / "b"), wrapup=StubWrapUp()), str(tmp_path)
    )
    assert "wrap_up" not in off._system() and '"type": "wrap_up"' in on._system()
    assert on._system().replace(
        on._system()[on._system().index("\n (c) ") : on._system().index("\n\nHOW TO WORK")], ""
    ) == (off._system())


def test_independent_search_scores_each_proposal_from_the_seed(tmp_path):
    lm = FakeLM()
    opt = WeightSpaceOptimizer(
        math_metric,
        ScriptedLM(),
        passes_per_round=2,
        reflection_minibatch_size=2,
        log_dir=str(tmp_path / "run"),
        max_turns=6,
        search="independent",
        n_proposals=2,
    )
    out = opt.compile(MathSolver(lm, max_new_tokens=64), trainset=examples(4), valset=examples(4))
    res = out.detailed_results
    assert PATCH in out.patch_src and out.weights_changed == 16
    assert len(res.candidates) == 3 and res.parents == [[None], [0], [0]]
    assert res.val_aggregate_scores[0] == 0.0 and res.val_aggregate_scores[1] == 1.0
    assert res.total_metric_calls == 4 + 2 * (2 + 4)
    assert res.candidates[0] == {COMPONENT: ""}
    json.dumps(res.candidates)  # run scripts write these to disk
    with pytest.raises(ValueError):
        WeightSpaceOptimizer(math_metric, ScriptedLM(), search="random")
    assert isinstance(examples(1)[0], dspy.Example)
