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


class PackedVLLM:
    """vLLM's layout for the projections it stacks: gate over up, and q over k over v."""

    def __init__(self):
        self.model = nn.Module()
        lang = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = nn.Linear(4, 12, bias=False)  # gate 6 rows, up 6 rows
        layer.mlp.down_proj = nn.Linear(6, 4, bias=False)
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = nn.Linear(4, 8, bias=False)  # q 4 rows, k 2 rows, v 2 rows
        lang.layers = nn.ModuleList([layer])
        self.model.language_model = lang

    def apply_model(self, fn):
        return [fn(self.model)]


def test_set_weights_places_a_part_in_its_rows_of_a_packed_matrix():
    backend = object.__new__(VLLMBackend)
    backend.llm = PackedVLLM()
    layer = backend.llm.model.language_model.layers[0]
    before = layer.mlp.gate_up_proj.weight.detach().clone()
    backend.set_weights({"model.layers.0.mlp.up_proj.weight": torch.full((6, 4), 2.0)})
    w = layer.mlp.gate_up_proj.weight
    assert torch.equal(w[:6], before[:6]) and torch.all(w[6:] == 2.0)
    backend.set_weights({"model.layers.0.mlp.gate_proj.weight": torch.full((6, 4), 3.0)})
    assert torch.all(w[:6] == 3.0) and torch.all(w[6:] == 2.0)
    for part, rows, where in (("q_proj", 4, slice(0, 4)), ("k_proj", 2, slice(4, 6)), ("v_proj", 2, slice(6, 8))):
        backend.set_weights(
            {f"model.layers.0.self_attn.{part}.weight": torch.full((rows, 4), float(rows + where.start))}
        )
        assert torch.all(layer.self_attn.qkv_proj.weight[where] == float(rows + where.start))


def test_a_patch_the_backend_cannot_place_changes_nothing_and_leaves_the_model_usable():
    backend = object.__new__(VLLMBackend)
    backend.llm = PackedVLLM()
    down = backend.llm.model.language_model.layers[0].mlp.down_proj.weight
    before = down.detach().clone()
    with pytest.raises(KeyError):
        backend.set_weights(
            {
                "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 6),
                "model.layers.0.mlp.nope.weight": torch.zeros(1, 1),
            }
        )
    assert torch.equal(down, before)  # nothing copied when one name fails

    class Refusing:
        def set_weights(self, tensors):
            raise KeyError("no")

    lm = FakeLM()
    lm.gen_backend = Refusing()
    with pytest.raises(KeyError):
        lm.apply_patch({L1: lm.original(L1) + 1.0})
    assert lm.edited() == [] and lm.changed_entries() == 0
    lm.reset()  # must not raise: the failed patch left nothing to undo
