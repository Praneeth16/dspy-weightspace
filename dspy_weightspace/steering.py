"""No-LLM baseline: a difference-in-means loop direction, applied in the edit family the optimizer found.

The optimizer's edits project a direction u out of a layer's MLP output: W' = (I - a*u*u^T) W on down_proj. Here u
comes from plain statistics instead of a reflection LM: mean residual state over the tail of looping traces minus the
mean over non-looping ones, as in difference-in-means steering. Directions are keyed by hidden-state index; the
down_proj of decoder layer L writes into hidden state L + 1.
"""

from __future__ import annotations

import torch

from .data import to_chat
from .hf_lm import HFLocalLM
from .metrics import LOOP_THRESHOLD, repetition
from .program import MathSolver


def down_proj(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj.weight"


def loop_directions(
    lm: HFLocalLM,
    prog: MathSolver,
    examples,
    hidden_layers: list[int],
    max_new: int = 4096,
    samples: int = 4,
    last_tokens: int = 1024,
) -> tuple[dict[int, torch.Tensor], dict]:
    """Sample base traces, label each looping or not with the metric's own detector, and return unit directions
    {hidden index: u} plus class counts. With only one class present, the more repetitive half (by rank, so ties
    cannot empty a class) counts as looping."""
    chats, texts, ratios = [], [], []
    for s in range(samples):
        preds = prog.batch_forward([e.problem for e in examples], seed=1000 * s, max_new_tokens=max_new)
        for e, p in zip(examples, preds, strict=True):
            chats.append(to_chat(e.problem))
            texts.append(p.text)
            ratios.append(repetition(p.text)[0])
    r = torch.tensor(ratios)
    looping = r < LOOP_THRESHOLD
    split = "threshold"
    if looping.all() or not looping.any():
        if len(r) < 2:
            raise ValueError("need at least two traces to contrast")
        looping, split = torch.zeros(len(r), dtype=torch.bool), "rank"
        looping[r.argsort()[: len(r) // 2]] = True
    hs = lm.hidden_states(chats, hidden_layers, texts, max_completion_tokens=max_new, chunk=2, last_tokens=last_tokens)
    dirs = {}
    for layer, h in hs.items():
        d = h[looping].mean(0) - h[~looping].mean(0)
        if not torch.isfinite(d).all() or d.norm() == 0:
            raise ValueError(f"degenerate direction at hidden state {layer}")
        dirs[layer] = d / d.norm()
    return dirs, {"looping": int(looping.sum()), "productive": int((~looping).sum()), "split": split}


def ablation_edits(lm: HFLocalLM, dirs: dict[int, torch.Tensor], layers: list[int], alpha: float) -> dict:
    """{down_proj of layer L: (I - alpha*u*u^T) W}, u = dirs[L + 1]. alpha 1 removes the direction, 2 reflects it."""
    edits = {}
    for layer in layers:
        name = lm.resolve(down_proj(layer))
        w = lm.original(name)
        u = dirs[layer + 1].float()
        u = u / u.norm()
        edits[name] = w - alpha * torch.outer(u, u @ w)
    return edits


def random_like(dirs: dict[int, torch.Tensor], seed: int = 0) -> dict[int, torch.Tensor]:
    """Unit directions drawn at random, same keys and size: the control for "any direction would do"."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer, u in dirs.items():
        v = torch.randn(u.shape, generator=g)
        out[layer] = v / v.norm()
    return out


def ablation_script(dirs: dict[int, torch.Tensor], layers: list[int], alpha: float, prefix: str) -> tuple[str, dict]:
    """The same edit as a sandbox patch plus the artifacts it loads, so the eval runner can replay it."""
    artifacts = {f"{prefix}_u_L{layer + 1}": dirs[layer + 1] for layer in layers}
    src = (
        f"# {prefix}: project a difference-in-means direction out of down_proj outputs, alpha={alpha:g}\n"
        f"alpha = {alpha!r}\n"
        f"for L in {list(layers)!r}:\n"
        "    n = f'model.layers.{L}.mlp.down_proj.weight'\n"
        "    W = load_original(n)\n"
        f"    u = load_artifact(f'{prefix}_u_L{{L + 1}}').float()\n"
        "    u = u / u.norm()\n"
        "    save_edit(n, W - alpha * torch.outer(u, u @ W))\n"
    )
    return src, artifacts
