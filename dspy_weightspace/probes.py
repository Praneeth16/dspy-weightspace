"""Forward-pass tool with a budget, as in the blog: ``generate`` and ``teacher_forced``.

Optionally a third request, ``wrap_up``: a measurement on fixed traces that needs no generation (see wrapup.py).
"""

from __future__ import annotations

import json
import os
from statistics import mean

import dspy
import numpy as np

from .data import to_chat
from .hf_lm import HFLocalLM
from .program import MathSolver
from .sandbox import PatchSandbox


class BudgetExhausted(RuntimeError):
    pass


class Probes:
    def __init__(
        self,
        lm: HFLocalLM,
        program: MathSolver,
        metric,
        practice: list[dspy.Example],
        work_dir: str,
        budget: int = 8,
        max_items: int = 32,
        wrapup=None,
        wrapup_cap: int = 24,
    ):
        self.lm, self.program, self.metric = lm, program, metric
        self.wrapup, self.wrapup_cap, self.wrapup_used = wrapup, wrapup_cap, 0
        self.practice = {f"p{i:03d}": ex for i, ex in enumerate(practice)}
        self.work_dir, self.budget, self.max_items = work_dir, budget, max_items
        self.used = 0
        self.sandbox: PatchSandbox | None = None
        os.makedirs(work_dir, exist_ok=True)

    @property
    def left(self) -> int:
        return self.budget - self.used

    def _spend(self) -> int:
        if self.left <= 0:
            raise BudgetExhausted("forward-pass budget exhausted")
        self.used += 1
        return self.used

    def _set_model(self, which: str) -> None:
        self.lm.reset()
        if which == "edited":
            if self.sandbox is None or not self.sandbox.edits:
                raise ValueError("no edits saved yet; use save_edit first")
            self.sandbox.apply()

    def catalogue(self) -> str:
        rows = [f"{k}: {v.problem[:80]!r} -> {v.answer}" for k, v in list(self.practice.items())[:12]]
        return (
            f"{len(self.practice)} practice problems with ids p000..p{len(self.practice) - 1:03d}. First 12:\n"
            + "\n".join(rows)
        )

    # ----- tools ------------------------------------------------------------------------------------
    def generate(self, req: dict) -> str:
        which = req.get("model", "original")
        ids = req.get("problems") or []
        prompts = req.get("prompts") or []
        n = int(req.get("n", 1))
        if not ids and not prompts:
            ids = list(self.practice)[: self.max_items]
        if (len(ids) + len(prompts)) * n > self.max_items:
            return f"error: items x n must be <= {self.max_items} (got {(len(ids) + len(prompts)) * n})"
        unknown = [i for i in ids if i not in self.practice]
        if unknown:
            return f"error: unknown problem ids {unknown[:5]}"
        k = self._spend()
        items = [(i, self.practice[i].problem, self.practice[i].answer) for i in ids] + [
            (f"prompt{j}", p, None) for j, p in enumerate(prompts)
        ]
        items = [it for it in items for _ in range(n)]
        max_tokens = int(req.get("max_tokens") or self.program.max_new_tokens or self.lm.kwargs["max_tokens"])
        layers = (req.get("activations") or {}).get("layers")

        results = {}
        for model in ["original", "edited"] if which == "both" else [which]:
            self._set_model(model)
            preds = self.program.batch_forward([p for _, p, _ in items], seed=0, max_new_tokens=max_tokens)
            rows = []
            for (pid, _problem, gold), pred in zip(items, preds, strict=True):
                m = self.metric(dspy.Example(answer=gold if gold is not None else ""), pred)
                rows.append(
                    {
                        "id": pid,
                        "model": model,
                        "finish_reason": pred.finish_reason,
                        "n_tokens": pred.n_tokens,
                        "correct": bool(m.correct) if gold is not None else None,
                        "looping": bool(m.looping),
                        "repetition_ratio": round(float(m.repetition_ratio), 3),
                        "text": pred.text,
                    }
                )
            if layers:
                hs = self.lm.hidden_states([to_chat(p) for _, p, _ in items], layers, [pr.text for pr in preds])
                for layer, t in hs.items():
                    np.save(os.path.join(self.work_dir, f"gen_{k}_{model}_L{layer}.npy"), t.numpy())
            results[model] = rows
            with open(os.path.join(self.work_dir, f"gen_{k}_{model}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        self.lm.reset()
        return self._summarize(k, results, bool(layers), [la for la in (layers or [])])

    def _summarize(self, k: int, results: dict, acts: bool, layers: list[int]) -> str:
        out = [f"pass {k} ({self.left} left)"]
        for model, rows in results.items():
            graded = [r for r in rows if r["correct"] is not None]
            acc = mean(r["correct"] for r in graded) if graded else float("nan")
            cut = mean(r["finish_reason"] == "length" for r in rows)
            loop = mean(r["looping"] for r in rows)
            toks = mean(r["n_tokens"] for r in rows)
            out.append(
                f"{model}: n={len(rows)} accuracy={acc:.3f} looping={loop:.3f} "
                f"cut_at_limit={cut:.3f} mean_tokens={toks:.0f}"
            )
            marks = []
            for r in rows:
                g = "-" if r["correct"] is None else ("C" if r["correct"] else "W")
                flags = ("/loop" if r["looping"] else "") + ("/cut" if r["finish_reason"] == "length" else "")
                marks.append(f"{r['id']}:{g}{flags}")
            out.append("  per item: " + " ".join(marks))
            out.append(
                f"  texts: {self.work_dir}/gen_{k}_{model}.jsonl "
                "(fields id, finish_reason, n_tokens, correct, looping, repetition_ratio, text)"
            )
        if len(results) == 2:
            o, e = results["original"], results["edited"]
            flips = [(a["correct"], b["correct"]) for a, b in zip(o, e, strict=True) if a["correct"] is not None]
            wr = sum(1 for a, b in flips if not a and b)
            rw = sum(1 for a, b in flips if a and not b)
            out.append(f"paired: wrong->right {wr}, right->wrong {rw}, diff {(wr - rw) / max(1, len(flips)):+.3f}")
        if acts:
            out.append(
                f"activations: {self.work_dir}/gen_{k}_<model>_L<layer>.npy for layers {layers}; "
                "each is [n_items, hidden], mean residual over generated tokens; row order = per item order above. "
                "Load with np.load, or in the kernel via torch.from_numpy(np.load(path))."
            )
        return "\n".join(out)

    def teacher_forced(self, req: dict) -> str:
        which = req.get("model", "original")
        chats = req.get("chats")
        texts = req.get("texts")
        layers = (req.get("return") or {}).get("hidden_states") or []
        if not layers:
            return "error: return.hidden_states (list of layer indices) is required"
        if chats:
            msgs = [[m for m in c if m["role"] == "user"] for c in chats]
            comps = ["".join(m["content"] for m in c if m["role"] == "assistant") for c in chats]
        elif texts:
            msgs = [[{"role": "user", "content": ""}] for _ in texts]
            comps = list(texts)
        else:
            return "error: provide chats or texts"
        if len(msgs) > self.max_items:
            return f"error: at most {self.max_items} texts"
        k = self._spend()
        self._set_model(which)
        hs = self.lm.hidden_states(msgs, layers, comps)
        self.lm.reset()
        paths = []
        for layer, t in hs.items():
            p = os.path.join(self.work_dir, f"fwd_{k}_{which}_L{layer}.npy")
            np.save(p, t.numpy())
            paths.append(p)
        return (
            f"pass {k} ({self.left} left): teacher-forced {len(msgs)} texts on {which}; "
            f"mean residual over assistant tokens saved: {paths} (each [n, hidden])"
        )

    def wrap_up(self, req: dict) -> str:
        """Not charged to the forward-pass budget: no generation, one teacher-forced pass per cached trace."""
        if self.wrapup_used >= self.wrapup_cap:
            return f"error: wrap_up limit ({self.wrapup_cap} per session) reached"
        which = req.get("model", "edited")
        self._set_model(which)
        out = self.wrapup.summary()
        self.lm.reset()
        self.wrapup_used += 1
        return f"wrap_up on {which} ({self.wrapup_cap - self.wrapup_used} left): {out}"

    def dispatch(self, req: dict) -> str:
        """Never raise into the proposer: a failed probe is reported to the model and does not cost a pass."""
        used_before = self.used
        try:
            if req.get("type") == "generate":
                return self.generate(req)
            if req.get("type") == "teacher_forced":
                return self.teacher_forced(req)
            if req.get("type") == "wrap_up" and self.wrapup is not None:
                return self.wrap_up(req)
            return "error: type must be 'generate' or 'teacher_forced'"
        except BudgetExhausted as e:
            return f"error: {e}"
        except Exception as e:  # noqa: BLE001
            self.used = used_before
            self.lm.reset()
            if self.lm.device == "cuda":
                import torch

                torch.cuda.empty_cache()
            msg = str(e).split("\n")[0][:300]
            return f"error: probe failed ({type(e).__name__}: {msg}). Not charged. Use fewer items or shorter texts."
