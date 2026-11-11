"""What an edit does at a sentence end, measured on fixed traces with no generation.

The base model's own thinking traces are fed through the edited weights (teacher forcing). At every sentence end we
read the probability that the next token starts another round of checking ("Wait") or wraps up ("So", "</think>").
Edits that raised MATH-500 accuracy made the wrap-up tokens 4 to 7.5 times more likely; weak edits 1.2 to 1.3 times.
The measurement has no sampling noise and costs one forward pass per trace, so it can screen many edits.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import to_chat
from .metrics import LOOP_THRESHOLD, repetition

STARTERS = [" Wait", "Wait", " But", " Alternatively", " Hmm", " However", " Let", " Actually"]
CLOSERS = ["</think>", " So", " Therefore", " Thus", " Final", "**"]
THINK_CLOSE = "</think>"
SENTENCE_END = (".", "\n", "?", ":", "!")
LOOP_CHUNK = 128


def single_ids(tok, words: list[str]) -> dict[str, int]:
    """The words that are one token for this tokenizer, with their ids."""
    out = {}
    for w in words:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            out[w] = ids[0]
    return out


def loop_labels(tok, ids: list[int], chunk: int = LOOP_CHUNK) -> np.ndarray:
    """Per token: inside a looping stretch? Chunks of `chunk` tokens, judged on the preceding 1,024 tokens."""
    lab = np.zeros(len(ids), dtype=bool)
    for s in range(0, len(ids), chunk):
        window = tok.decode(ids[max(0, s - 1024) : s + chunk])
        lab[s : s + chunk] = repetition(window)[0] < LOOP_THRESHOLD
    return lab


def boundary_mask(tok, ids: list[int]) -> np.ndarray:
    """Per token: does it come right after a sentence end or a line break?"""
    prev = [tok.decode([i]) for i in ids]
    return np.array([k > 0 and prev[k - 1].rstrip(" ").endswith(SENTENCE_END) for k in range(len(ids))])


class WrapUpProbe:
    """Compares the current weights of `lm` with its original weights on fixed (problem, completion) traces."""

    def __init__(self, lm, traces: list[tuple[str, str]], max_tokens: int = 6144, chunk: int = 512):
        self.lm, self.chunk = lm, chunk
        tok = lm.tok
        self.groups = {
            "starters": list(single_ids(tok, STARTERS).values()),
            "closers": list(single_ids(tok, CLOSERS).values()),
        }
        self.think_id = single_ids(tok, [THINK_CLOSE]).get(THINK_CLOSE)
        self.rows = []
        for problem, text in traces:
            p = tok(lm._render(to_chat(problem)), add_special_tokens=False)["input_ids"]
            c = tok(text, add_special_tokens=False)["input_ids"][:max_tokens]
            if len(c) < 2:
                continue
            self.rows.append({"p": p, "c": c, "loop": loop_labels(tok, c), "boundary": boundary_mask(tok, c)})
        if not self.rows or not any(r["boundary"].any() for r in self.rows):
            raise ValueError("wrap-up probe needs at least one trace with a sentence end")
        if getattr(lm, "_edited", None):
            raise ValueError("build the wrap-up probe on the original weights")
        self._base = [self._hidden(r) for r in self.rows]  # final hidden states under the original weights

    @torch.no_grad()
    def _hidden(self, r: dict) -> torch.Tensor:
        ids = torch.tensor([r["p"] + r["c"]], device=self.lm.device)
        n_p = len(r["p"])
        # Decoder only: full-vocabulary logits for thousands of positions at once would not fit in memory.
        h = self.lm.hf.base_model(input_ids=ids).last_hidden_state[0]
        return h[n_p - 1 : n_p - 1 + len(r["c"])]  # the states that predict each completion token

    @torch.no_grad()
    def measure(self) -> dict[str, float]:
        """closers_x / starters_x: probability mass at sentence ends, relative to the original weights (1.0 = same).
        think_close_dlogp: change in log-probability of the closing think tag at sentence ends.
        kl_loop / kl_productive: mean KL(original || current) on looping and on productive tokens."""
        head = self.lm.hf.get_output_embeddings()
        mass = {k: [] for k in ("starters_b", "starters_e", "closers_b", "closers_e")}
        kls, loops, think = [], [], []
        for r, hb in zip(self.rows, self._base, strict=True):
            he = self._hidden(r)
            bnd = torch.from_numpy(r["boundary"])
            for c0 in range(0, hb.shape[0], self.chunk):
                lb = torch.log_softmax(head(hb[c0 : c0 + self.chunk]).float(), -1)
                le = torch.log_softmax(head(he[c0 : c0 + self.chunk]).float(), -1)
                kls.append((lb.exp() * (lb - le)).sum(-1).cpu())
                b = bnd[c0 : c0 + self.chunk]
                for g, gids in self.groups.items():
                    mass[f"{g}_b"].append(torch.logsumexp(lb[:, gids], -1).exp().cpu()[b])
                    mass[f"{g}_e"].append(torch.logsumexp(le[:, gids], -1).exp().cpu()[b])
                if self.think_id is not None:
                    think.append((le[:, self.think_id] - lb[:, self.think_id]).cpu()[b])
            loops.append(torch.from_numpy(r["loop"]))
        kl, loop = torch.cat(kls), torch.cat(loops)
        m = {k: float(torch.cat(v).mean()) for k, v in mass.items()}
        return {
            "closers_x": m["closers_e"] / max(m["closers_b"], 1e-12),
            "starters_x": m["starters_e"] / max(m["starters_b"], 1e-12),
            "closers_p": m["closers_e"],
            "starters_p": m["starters_e"],
            "think_close_dlogp": float(torch.cat(think).mean()) if think else 0.0,
            "kl_loop": float(kl[loop].mean()) if loop.any() else 0.0,
            "kl_productive": float(kl[~loop].mean()) if (~loop).any() else 0.0,
        }

    def summary(self) -> str:
        m = self.measure()
        return (
            f"closers_x={m['closers_x']:.2f} starters_x={m['starters_x']:.2f} "
            f"think_close_dlogp={m['think_close_dlogp']:+.2f} kl_loop={m['kl_loop']:.3f} "
            f"kl_productive={m['kl_productive']:.3f}"
        )
