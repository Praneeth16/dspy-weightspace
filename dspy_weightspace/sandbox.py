"""Persistent Python kernel the reflection LM uses to inspect and edit weights.

Tool surface mirrors the blog's harness so transcripts are comparable: ``NAMES``, ``load``,
``load_original``, ``save_edit``, ``drop_edit``, ``edited``, ``tok``. Extras: ``save_artifact`` /
``load_artifact`` so a final patch script can reference vectors found during exploration.
"""

from __future__ import annotations

import contextlib
import io
import os
import traceback
from dataclasses import dataclass, field

import numpy as np
import torch

from .hf_lm import HFLocalLM, is_editable

MAX_OUTPUT_CHARS = 10_000


class EditRejected(ValueError):
    pass


@dataclass
class ExecResult:
    stdout: str
    error: str | None = None
    edits: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


class PatchSandbox:
    def __init__(self, lm: HFLocalLM, work_dir: str):
        self.lm = lm
        self.work_dir = work_dir
        os.makedirs(os.path.join(work_dir, "artifacts"), exist_ok=True)
        self.edits: dict[str, torch.Tensor] = {}
        self.ns: dict = {}
        self._reset_namespace()

    def _reset_namespace(self) -> None:
        self.ns = {
            "NAMES": list(self.lm.names),
            "load": self.load,
            "load_original": self.lm.original,
            "save_edit": self.save_edit,
            "drop_edit": self.drop_edit,
            "edited": lambda: sorted(self.edits),
            "save_artifact": self.save_artifact,
            "load_artifact": self.load_artifact,
            "tok": self.lm.tok,
            "torch": torch,
            "np": np,
            "WORK": self.work_dir,
        }

    # ----- tools ------------------------------------------------------------------------------------
    def load(self, name: str) -> torch.Tensor:
        name = self.lm.resolve(name)
        if name in self.edits:
            return self.edits[name].clone()
        return self.lm.original(name)

    def save_edit(self, name: str, tensor) -> None:
        try:
            name = self.lm.resolve(name)
        except KeyError:
            raise EditRejected(f"unknown tensor {name}") from None
        if not is_editable(name):
            raise EditRejected(f"{name} is out of scope (embeddings, output head and final norm are frozen)")
        t = torch.as_tensor(tensor).detach().float().cpu()
        base = self.lm.original(name)
        if tuple(t.shape) != tuple(base.shape):
            raise EditRejected(f"{name}: shape {tuple(t.shape)} != {tuple(base.shape)}")
        if not torch.isfinite(t).all():
            raise EditRejected(f"{name}: non-finite values")
        if _is_scalar_multiple(t, base):
            raise EditRejected(f"{name}: multiplying a whole tensor by a constant is out of scope")
        self.edits[name] = t

    def drop_edit(self, name: str) -> None:
        self.edits.pop(self.lm.resolve(name), None)

    def save_artifact(self, name: str, tensor) -> str:
        path = os.path.join(self.work_dir, "artifacts", f"{name}.npy")
        np.save(path, torch.as_tensor(tensor).detach().float().cpu().numpy())
        return path

    def load_artifact(self, name: str) -> torch.Tensor:
        return torch.from_numpy(np.load(os.path.join(self.work_dir, "artifacts", f"{name}.npy")))

    # ----- execution --------------------------------------------------------------------------------
    def run(self, src: str) -> ExecResult:
        buf = io.StringIO()
        err = None
        with torch.no_grad(), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(src, "<patch>", "exec"), self.ns)
            except Exception:
                err = traceback.format_exc(limit=3)
        out = buf.getvalue()
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... [cut at {MAX_OUTPUT_CHARS} chars]"
        return ExecResult(stdout=out, error=err, edits=sorted(self.edits))

    def apply(self) -> None:
        self.lm.reset()
        if self.edits:
            self.lm.apply_patch(self.edits)

    def clear(self) -> None:
        self.edits.clear()
        self.lm.reset()
        self._reset_namespace()

    def export_script(self) -> str:
        """Freeze the current edits as artifacts and return a standalone patch that reloads them."""
        lines = ["# Exported edits (fallback when the session ended without a ```patch block)"]
        for name in sorted(self.edits):
            key = "edit__" + name.replace(".", "_")
            self.save_artifact(key, self.edits[name])
            lines.append(f"save_edit({name!r}, load_artifact({key!r}))")
        return "\n".join(lines) + "\n"


def _is_scalar_multiple(t: torch.Tensor, base: torch.Tensor, tol: float = 1e-4) -> bool:
    if torch.equal(t, base):
        return False
    mask = base.abs() > 1e-6
    if mask.sum() < 16:
        return False
    ratio = t[mask] / base[mask]
    return bool(ratio.std() < tol * max(1.0, float(ratio.mean().abs())))


def run_fresh(lm: HFLocalLM, src: str, work_dir: str) -> tuple[PatchSandbox, ExecResult]:
    """Run a standalone patch script against original weights in a new sandbox (used per GEPA candidate)."""
    sb = PatchSandbox(lm, work_dir)
    res = sb.run(src) if src.strip() else ExecResult(stdout="")
    return sb, res
