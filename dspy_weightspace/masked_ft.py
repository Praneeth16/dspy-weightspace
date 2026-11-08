"""Gradient baseline on the weights the optimizer is allowed to touch: masked supervised fine-tuning.

Only entries inside a mask move (for example the 131,072 the published edit changed). Two training sets: the base
model's own correct, terminated traces on the feedback problems (the information the optimizer sees) and GSM8K's
written rationales (more information, an upper reference).
"""

from __future__ import annotations

import random

import torch

from .data import to_chat
from .hf_lm import HFLocalLM
from .metrics import extract_answer, is_correct
from .program import MathSolver


def support_mask(lm: HFLocalLM, which: str, published: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """ "published": the entries the published edit changed. "layers": every entry of those tensors."""
    masks = {}
    for name, edited in published.items():
        name = lm.resolve(name)
        base = lm.original(name)
        masks[name] = (edited != base) if which == "published" else torch.ones_like(base, dtype=torch.bool)
    return masks


def self_traces(prog: MathSolver, examples, samples: int, max_new: int) -> list[tuple[str, str]]:
    """Rejection sampling: base completions that finished and got the answer right."""
    keep = []
    for s in range(samples):
        preds = prog.batch_forward([e.problem for e in examples], seed=7000 + 1000 * s, max_new_tokens=max_new)
        for e, p in zip(examples, preds, strict=True):
            if p.finish_reason != "length" and is_correct(extract_answer(p.text), e.answer):
                keep.append((e.problem, p.text))
    return keep


def gold_traces(examples) -> list[tuple[str, str]]:
    return [(e.problem, f"{e.rationale}\n</think>\n\n\\boxed{{{e.answer}}}") for e in examples]


def masked_sft(
    lm: HFLocalLM,
    pairs: list[tuple[str, str]],
    masks: dict[str, torch.Tensor],
    lr: float = 1e-4,
    checkpoints: tuple[int, ...] = (25, 50, 100, 200),
    accumulate: int = 4,
    max_len: int = 8192,
    seed: int = 0,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[float]]:
    """AdamW (no weight decay, so unmasked entries never move) on float32 copies of the masked tensors; loss on the
    completion tokens only. Returns ({step: {name: tensor}}, loss per step); the model is restored afterwards."""
    params = {n: lm._params[n] for n in masks}
    master = {n: lm.original(n).to(lm.device).requires_grad_(True) for n in masks}
    dev_masks = {n: m.to(lm.device) for n, m in masks.items()}
    opt = torch.optim.AdamW(list(master.values()), lr=lr, weight_decay=0.0)
    eos = lm.tok.eos_token or ""
    rows = []
    for problem, completion in pairs:
        prompt = lm._render(to_chat(problem))
        p_ids = lm.tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = lm.tok(completion + eos, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[:max_len]
        labels = ([-100] * len(p_ids) + c_ids)[:max_len]
        rows.append((torch.tensor([ids]), torch.tensor([labels])))
    rng = random.Random(seed)
    snapshots, losses = {}, []
    try:
        for p in params.values():
            p.requires_grad_(True)
        for step in range(1, max(checkpoints) + 1):
            opt.zero_grad()
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(master[n].to(p.dtype))
            total = 0.0
            for ids, labels in rng.sample(rows, min(accumulate, len(rows))):
                with torch.enable_grad():
                    out = lm.hf(input_ids=ids.to(lm.device), labels=labels.to(lm.device))
                    (out.loss / accumulate).backward()
                total += float(out.loss.detach()) / accumulate
            for n, p in params.items():
                master[n].grad = (p.grad.float() * dev_masks[n]) if p.grad is not None else torch.zeros_like(master[n])
                p.grad = None
            opt.step()
            losses.append(total)
            if step in checkpoints:
                snapshots[step] = {n: master[n].detach().cpu().clone() for n in masks}
    finally:
        with torch.no_grad():
            for n, p in params.items():
                p.requires_grad_(False)
                p.grad = None
                p.copy_(lm._base[n].to(p.device))
    return snapshots, losses
