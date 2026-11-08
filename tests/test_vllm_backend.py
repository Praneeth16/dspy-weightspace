from __future__ import annotations

import pytest
import torch
from torch import nn

from dspy_weightspace.hf_lm import Completion, HFLocalLM
from dspy_weightspace.metrics import boxed_all, extract_answer, is_correct
from dspy_weightspace.vllm_backend import VLLMBackend, layer_suffix

from .conftest import FakeLM

L1 = "model.language_model.layers.1.mlp.down_proj.weight"


class RecordingBackend:
    def __init__(self):
        self.pushed, self.prompts = [], []

    def set_weights(self, tensors):
        self.pushed.append({n: t.clone() for n, t in tensors.items()})

    def generate(self, prompts, max_new_tokens, temperature=None, seed=None):
        self.prompts += prompts
        return [Completion("ok</think> \\boxed{1}", "stop", 3) for _ in prompts]


def test_patch_and_reset_are_mirrored_and_generation_goes_to_the_backend():
    lm = FakeLM()
    lm.gen_backend = RecordingBackend()
    new = lm.original(L1) + 1.0
    lm.apply_patch({L1: new})
    assert torch.equal(lm.gen_backend.pushed[-1][L1], new)
    lm.reset()
    assert torch.equal(lm.gen_backend.pushed[-1][L1], lm._base[L1])
    lm.reset()  # nothing edited: nothing pushed
    assert len(lm.gen_backend.pushed) == 2
    out = HFLocalLM.generate_detailed(lm, [[{"role": "user", "content": "q"}]], max_new_tokens=8)
    assert out[0].finish_reason == "stop" and lm.gen_backend.prompts == ["q"]


class FakeVLLM:
    """apply_model runs the function on a module named the way vLLM names Qwen3.5's language model."""

    def __init__(self):
        self.model = nn.Module()
        lang = nn.Module()
        lang.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in lang.layers:
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(16, 8, bias=False)
        self.model.language_model = lang

    def apply_model(self, fn):
        return [fn(self.model)]


def test_set_weights_copies_into_the_matching_vllm_parameter():
    backend = object.__new__(VLLMBackend)
    backend.llm = FakeVLLM()
    t = torch.full((8, 16), 0.5)
    backend.set_weights({"model.layers.1.mlp.down_proj.weight": t})
    got = backend.llm.model.language_model.layers[1].mlp.down_proj.weight
    assert torch.allclose(got, t) and not torch.allclose(
        backend.llm.model.language_model.layers[0].mlp.down_proj.weight, t
    )
    with pytest.raises(ValueError):
        backend.set_weights({"model.layers.0.mlp.down_proj.weight": torch.zeros(3, 3)})
    with pytest.raises(KeyError):
        backend.set_weights({"model.layers.7.mlp.down_proj.weight": t})
    assert layer_suffix("model.language_model.layers.15.mlp.down_proj.weight") == "layers.15.mlp.down_proj.weight"


def test_nested_boxed_and_symbolic_answers():
    assert boxed_all(r"x \boxed{\frac{\sqrt{3}}{2}} y \boxed{7} \boxed{open") == [r"\frac{\sqrt{3}}{2}", "7"]
    assert extract_answer(r"<think>t</think> so \boxed{\dfrac{1}{2}}") == r"\dfrac{1}{2}"
    pytest.importorskip("math_verify")
    assert is_correct(r"\dfrac{1}{2}", "0.5") and not is_correct(r"\frac{1}{3}", "0.5")
