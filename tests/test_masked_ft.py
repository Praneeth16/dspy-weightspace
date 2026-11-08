from __future__ import annotations

import pytest
import torch

from dspy_weightspace.masked_ft import gold_traces, masked_sft, support_mask

transformers = pytest.importorskip("transformers")


class CharTok:
    eos_token = "|"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 64 for c in text]}


class TinyLM:
    """Just what masked_sft touches: a real 1-layer GPT-2 (no download), its params, base copies and a renderer."""

    def __init__(self):
        cfg = transformers.GPT2Config(vocab_size=64, n_positions=256, n_embd=16, n_layer=1, n_head=2)
        torch.manual_seed(0)
        self.hf = transformers.GPT2LMHeadModel(cfg).eval()
        for p in self.hf.parameters():
            p.requires_grad_(False)
        self._params = dict(self.hf.named_parameters())
        self._base = {n: p.detach().clone() for n, p in self._params.items()}
        self.device, self.tok = "cpu", CharTok()

    def resolve(self, name):
        return name

    def original(self, name):
        return self._base[name].clone().float()

    def _render(self, chat):
        return chat[0]["content"]


W = "transformer.h.0.mlp.c_proj.weight"


def test_only_masked_entries_move_and_loss_drops():
    lm = TinyLM()
    base = lm.original(W)
    published = {W: base.clone()}
    published[W][:, :3] += 1.0  # pretend the published edit changed these 3 columns
    masks = support_mask(lm, "published", published)
    assert int(masks[W].sum()) == base.shape[0] * 3
    pairs = [("2+2=", "four, so 4"), ("3+3=", "six, so 6")]
    snaps, losses = masked_sft(lm, pairs, masks, lr=1e-2, checkpoints=(5, 20), accumulate=2)
    moved = snaps[20][W] != base
    assert moved.any() and not (moved & ~masks[W]).any()
    assert losses[-1] < losses[0]
    assert torch.equal(lm._params[W], lm._base[W]) and not lm._params[W].requires_grad


def test_gold_trace_format():
    e = type("E", (), {"problem": "q", "rationale": "48/2 = 24", "answer": "72"})()
    assert gold_traces([e]) == [("q", "48/2 = 24\n</think>\n\n\\boxed{72}")]
