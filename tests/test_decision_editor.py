import torch

from dspy_weightspace.decision_editor import LAYER_SETS, Decision, EditBuilder, strength_multiplier

from .conftest import FakeLM


class FakeLM2(FakeLM):
    """Two layers named 0 and 1 plus embeddings; wire layer sets onto them."""

    def resolve(self, name):
        name = name.replace("model.language_model.", "model.")
        for la in (15, 19, 13, 14, 16, 17, 10, 11, 12, 18, 20, 21, 22, 23):
            name = name.replace(f"layers.{la}.", f"layers.{la % 2}.")
        return super().resolve(name)

    def hidden_states(self, chats, layers, completions=None):
        self.passes += 1
        out = {}
        for layer in layers:
            rows = torch.zeros(len(chats), 8)
            if completions and "Wait" in completions[0]:
                rows[:, 0] = 1.0
            out[layer] = rows
        return out


def test_strength_grid():
    assert strength_multiplier(0) == 0.5
    assert strength_multiplier(1) == 1.0
    assert abs(strength_multiplier(2.5) - 2.0) < 1e-9
    assert strength_multiplier(4) == 3.2
    assert strength_multiplier(9) == 3.2


def test_project_out_touches_only_chosen_layers():
    lm = FakeLM2()
    b = EditBuilder(lm)
    d = Decision("project_out", "15_19", 1.0, True, "", {})
    edits = b.build(d)
    key = lm.resolve("model.layers.1.mlp.down_proj.weight")  # 15 and 19 both map to layer 1 in the fake
    assert set(edits) == {key}
    assert not torch.equal(edits[key], lm.original(key))


def test_no_ship_means_no_edits():
    b = EditBuilder(FakeLM2())
    assert b.build(Decision("project_out", "15_19", 1.0, False, "", {})) == {}


def test_scale_published_interpolates():
    lm = FakeLM2()
    name = "model.layers.1.mlp.down_proj.weight"
    w0 = lm.original(name)
    pub = w0 + 1.0
    b = EditBuilder(lm, published={name: pub})
    e = b.build(Decision("scale_published", "15_19", 2.0, True, "", {}))
    assert torch.allclose(e[lm.resolve(name)], w0 + 2.0)


def test_layer_sets_cover_expected_depths():
    assert LAYER_SETS["15_19"] == [15, 19] and len(LAYER_SETS["10_21"]) == 12
