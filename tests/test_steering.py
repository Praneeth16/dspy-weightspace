from __future__ import annotations

import torch

from dspy_weightspace.program import MathSolver
from dspy_weightspace.sandbox import run_fresh
from dspy_weightspace.steering import ablation_edits, ablation_script, loop_directions, random_like

from .conftest import FakeLM

L1 = "model.language_model.layers.1.mlp.down_proj.weight"


def unit(n, seed):
    v = torch.randn(n, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def test_weight_edit_equals_projecting_the_mlp_output():
    lm = FakeLM()
    u = unit(8, 1)
    for alpha in (1.0, 3.0):
        w2 = ablation_edits(lm, {2: u}, [1], alpha)[L1]
        x = torch.randn(5, 16, generator=torch.Generator().manual_seed(2))
        y = x @ lm.original(L1).T
        hooked = y - alpha * (y @ u)[:, None] * u[None, :]
        assert torch.allclose(x @ w2.T, hooked, atol=1e-5)
    removed = torch.randn(3, 16) @ ablation_edits(lm, {2: u}, [1], 1.0)[L1].T
    assert torch.allclose(removed @ u, torch.zeros(3), atol=1e-5)


def test_random_control_is_unit_and_differs():
    u = unit(8, 1)
    r = random_like({2: u}, seed=3)[2]
    assert abs(float(r.norm()) - 1.0) < 1e-6 and abs(float(r @ u)) < 0.99


def test_script_replays_to_the_same_edit(tmp_path):
    lm = FakeLM()
    dirs = {2: unit(8, 1)}
    src, artifacts = ablation_script(dirs, [1], 3.0, "diffmean")
    from dspy_weightspace.sandbox import PatchSandbox

    sb = PatchSandbox(lm, str(tmp_path))
    for k, v in artifacts.items():
        sb.save_artifact(k, v)
    fresh, res = run_fresh(lm, src, str(tmp_path))
    assert res.ok, res.error
    assert torch.allclose(fresh.edits[L1], ablation_edits(lm, dirs, [1], 3.0)[L1], atol=1e-6)


class StatesLM(FakeLM):
    """Looping traces sit at +1 on coordinate 0 of hidden state 2; the direction must point there."""

    def hidden_states(self, chats, layers, completions=None, **kw):
        h = torch.zeros(len(chats), 8)
        for i, c in enumerate(completions):
            h[i, 0] = 1.0 if "Wait" in c else 0.0
        return {layer: h.clone() for layer in layers}

    def generate_detailed(self, chats, max_new_tokens=None, temperature=None, seed=None, **kw):
        out = super().generate_detailed(chats, max_new_tokens, temperature, seed)
        if seed:  # second sampling round: productive traces
            self.loops = False
        return out


def test_loop_direction_points_from_productive_to_looping():
    lm = StatesLM()
    prog = MathSolver(lm, max_new_tokens=64)
    exs = [type("E", (), {"problem": f"p{i}"})() for i in range(3)]
    dirs, counts = loop_directions(lm, prog, exs, [2], samples=3)
    assert counts["looping"] > 0 and counts["productive"] > 0
    assert torch.allclose(dirs[2], torch.eye(8)[0], atol=1e-6)


class TiedLM(StatesLM):
    """Every trace loops identically: equal ratios must still give two non-empty classes."""

    def hidden_states(self, chats, layers, completions=None, **kw):
        h = torch.zeros(len(chats), 8)
        h[: len(chats) // 2, 0] = 1.0
        h[:, 1] = torch.arange(len(chats), dtype=torch.float32)
        return {layer: h.clone() for layer in layers}

    def generate_detailed(self, chats, max_new_tokens=None, temperature=None, seed=None, **kw):
        return FakeLM.generate_detailed(self, chats, max_new_tokens, temperature, seed)


def test_tied_ratios_still_split_into_two_classes():
    lm = TiedLM()
    exs = [type("E", (), {"problem": f"p{i}"})() for i in range(2)]
    dirs, counts = loop_directions(lm, MathSolver(lm, max_new_tokens=64), exs, [2], samples=2)
    assert counts == {"looping": 2, "productive": 2, "split": "rank"}
    assert torch.isfinite(dirs[2]).all()
