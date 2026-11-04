"""Track B: structured weight edits chosen with DSPy decision types (Jev), not free-form code.

The reflection model does not write code. It reads diagnostics and makes four decisions: which edit method, which
layers, how strong, and whether to ship. A deterministic builder turns the decisions into tensors. The search then
belongs to stock DSPy: ``dspy.GEPA`` tunes the decision instruction, ``ReAnchor`` calibrates the strength cuts and the
ship threshold against the metric. Every step is reproducible from the saved decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import dspy
import torch
from dspy.experimental import Choice, Noul, Score

from .data import to_chat
from .hf_lm import HFLocalLM

Method = Choice[
    (
        "project_out",
        "Remove the repetition direction u (difference of mean residuals, repetitive minus productive continuations) "
        "from the MLP output: W = W0 - 6*s*outer(u, mask64 * (u @ W0)) on the 64 down_proj columns writing most "
        "along u. This is the form of the published Experiment-8 edit, recomputed here.",
    ),
    (
        "scale_published",
        "Scale the published Experiment-8 delta itself: W = W0 + s*(W_published - W0) in layers 15 and 19. "
        "s=1 reproduces the published edit exactly. Ignores the layer choice.",
    ),
    (
        "damp_wait_neurons",
        "Shrink the 16 down_proj columns per layer whose output most increases the logit of 'Wait'-type tokens, "
        "by a factor (1 - 0.3*s). Targets the trigger of the re-read loop rather than its direction.",
    ),
]
Layers = Choice[
    ("15_19", "layers 15 and 19, where the published edit lives (full-attention layers)"),
    ("13_17", "layers 13 to 17, mid-depth"),
    ("10_21", "layers 10 to 21, wide"),
    ("20_23", "layers 20 to 23, late"),
]
Strength = Score["0.5x, a quarter of the published strength", "1.0x, the published strength", "1.6x", "2.4x", "3.2x"]
Ship = Noul[
    (True, "the evidence supports applying this edit: fewer loops AND at least as many correct answers"),
    (False, "no reliable gain, or accuracy at risk; keep the original weights"),
]
STRENGTHS = [0.5, 1.0, 1.6, 2.4, 3.2]
LAYER_SETS = {"15_19": [15, 19], "13_17": [13, 14, 15, 16, 17], "10_21": list(range(10, 22)), "20_23": [20, 21, 22, 23]}

REPETITIVE = [
    "Wait, let me re-read the question again. " * 30,
    "Hmm, let me check the phrasing once more. Actually, wait. Let me look at the wording again. " * 12,
    "Let me verify this one more time. Wait, I should re-read the problem. " * 15,
    "Wait, does the question say that? Let me re-read. Wait. " * 20,
]
PRODUCTIVE = [
    "First, 18 divided by 3 is 6, so 6 contestants are female. Then 18 minus 6 is 12 male contestants. \\boxed{12}",
    "Marco scored 10% below 90, which is 81. Margaret scored 5 more than Marco, so 86. \\boxed{86}",
    "Half of 8 slices is 4, and a quarter of 8 is 2. 8 minus 4 minus 2 leaves 2 slices. \\boxed{2}",
    "The price is 85. Twenty percent off gives 68, then 25 percent off gives 51. \\boxed{51}",
]


class DecideEdit(dspy.Signature):
    """Choose one weight edit for a small reasoning model that loops in its thinking until it hits the token limit.

    You are given diagnostics: how often the model loops and is cut off, its accuracy, what earlier edits did, and
    facts about the published edit. Pick the method, the layers and the strength most likely to raise the number of
    CORRECT final answers. Stopping the loops only matters because it lets answers finish; an edit that silences
    thinking without raising accuracy is a failure. Decide whether to ship at all: when the evidence is weak, keep the
    original weights."""

    diagnostics: str = dspy.InputField(desc="Loop rate, cut rate, accuracy, prior edits with their paired results.")
    method: Method = dspy.OutputField(desc="Which edit family to apply.")
    layers: Layers = dspy.OutputField(desc="Which layers to edit (ignored by scale_published).")
    strength: Strength = dspy.OutputField(desc="How strong, as a multiple of the published edit's strength.")
    ship: Ship = dspy.OutputField(desc="Apply the edit, or keep the original weights.")
    rationale: str = dspy.OutputField(desc="Two sentences: the mechanism you expect and the evidence for it.")


def strength_multiplier(value: float) -> float:
    """Map a Score value in [0, 4] onto the strength grid by linear interpolation."""
    v = min(max(float(value), 0.0), 4.0)
    i = min(int(v), 3)
    return STRENGTHS[i] + (v - i) * (STRENGTHS[i + 1] - STRENGTHS[i])


@dataclass
class Decision:
    method: str
    layers: str
    strength: float
    ship: bool
    rationale: str
    evidence: dict

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


class EditBuilder:
    """Turns a Decision into tensors. Directions are computed once per model and cached."""

    def __init__(self, lm: HFLocalLM, published: dict[str, torch.Tensor] | None = None):
        self.lm = lm
        self.published = published or {}
        self._u: dict[int, torch.Tensor] = {}
        self._wait: torch.Tensor | None = None

    def _layer_name(self, layer: int) -> str:
        return self.lm.resolve(f"model.layers.{layer}.mlp.down_proj.weight")

    def directions(self, layers: list[int]) -> dict[int, torch.Tensor]:
        missing = [la for la in layers if la not in self._u]
        if missing:
            chats = [
                to_chat(
                    "Nancy bought a pie sliced into 8 pieces. She gave 1/2 to Joe and 1/4 to Carl. How many are left?"
                )
            ]
            idx = [la + 1 for la in missing]  # hidden_states[l+1] is the residual after layer l
            rep = self.lm.hidden_states(chats * len(REPETITIVE), idx, REPETITIVE)
            good = self.lm.hidden_states(chats * len(PRODUCTIVE), idx, PRODUCTIVE)
            for la, i in zip(missing, idx, strict=True):
                d = rep[i].mean(0) - good[i].mean(0)
                self._u[la] = d / (d.norm() + 1e-8)
        return {la: self._u[la] for la in layers}

    def wait_direction(self) -> torch.Tensor:
        if self._wait is None:
            tok = self.lm.tok
            ids = [tok.encode(w, add_special_tokens=False)[0] for w in (" Wait", "Wait", " wait", " Hmm", " Actually")]
            emb = self.lm.original(self.lm.resolve("model.embed_tokens.weight"))
            d = emb[ids].mean(0)
            self._wait = d / (d.norm() + 1e-8)
        return self._wait

    def build(self, decision: Decision) -> dict[str, torch.Tensor]:
        if not decision.ship:
            return {}
        s = decision.strength
        edits: dict[str, torch.Tensor] = {}
        if decision.method == "scale_published":
            for name, w_pub in self.published.items():
                w0 = self.lm.original(name)
                edits[self.lm.resolve(name)] = w0 + s * (w_pub - w0)
            return edits
        layers = LAYER_SETS[decision.layers]
        if decision.method == "project_out":
            for la, u in self.directions(layers).items():
                name = self._layer_name(la)
                w0 = self.lm.original(name)
                writer = u @ w0
                mask = torch.zeros_like(writer)
                top = writer.abs().topk(min(64, writer.numel())).indices
                mask[top] = writer[top]
                edits[name] = w0 - 6.0 * s * torch.outer(u, mask)
            return edits
        if decision.method == "damp_wait_neurons":
            d = self.wait_direction()
            for la in layers:
                name = self._layer_name(la)
                w0 = self.lm.original(name)
                score = d @ w0
                top = score.topk(min(16, score.numel())).indices
                w = w0.clone()
                w[:, top] *= 1.0 - 0.3 * s
                edits[name] = w
            return edits
        raise ValueError(decision.method)


class DecisionEditor(dspy.Module):
    """diagnostics -> decisions -> weight edits. The LM never touches a tensor."""

    def __init__(self, builder: EditBuilder):
        super().__init__()
        self.decide = dspy.Predict(DecideEdit)
        self.builder = builder

    def forward(self, diagnostics: str) -> dspy.Prediction:
        r = self.decide(diagnostics=diagnostics)
        decision = Decision(
            method=r.method.value,
            layers=r.layers.value,
            strength=strength_multiplier(float(r.strength)),
            ship=bool(r.ship.value),
            rationale=r.rationale,
            evidence={
                "method": getattr(r.method, "probabilities", None),
                "strength": getattr(r.strength, "probabilities", None),
                "ship_probability": getattr(r.ship, "probability", None),
            },
        )
        edits = self.builder.build(decision)
        return dspy.Prediction(decision=decision, edits=edits, n_edited=len(edits))


def diagnostics_text(rows: dict[str, dict], extra: str = "") -> str:
    lines = ["Holdout-style measurements on practice problems (thinking on, fixed token budget):"]
    for k, v in rows.items():
        lines.append(
            f"- {k}: accuracy {v['accuracy']:.1%}, looping {v['loop_rate']:.1%}, cut at limit {v['cut_rate']:.1%}, "
            f"mean tokens {v['mean_tokens']:.0f}"
        )
    lines.append(
        "Facts: the published Experiment-8 edit changes 64 down_proj columns in layers 15 and 19 along one output "
        "direction per layer (rank-1); scaling it beyond about 2x removes loops but costs accuracy."
    )
    if extra:
        lines.append(extra)
    return "\n".join(lines)
