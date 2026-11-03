"""The edit Astra submitted in Experiment 8 (50-pass run), rebuilt from the public logs.

Source: github.com/RikiyaT/lm_understand-logs, ``exp8_math/runs/p50c/edit/`` (``manifest.json`` + ``edit.npz``,
form "sparse": flat indices and bfloat16 bit values for every changed entry). Two tensors:
``model.language_model.layers.{15,19}.mlp.down_proj.weight``. Rebuilt tensors are sha256-checked against the
checkpoint the blog evaluated, exactly as the authors' ``apply_edit.py`` does.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import numpy as np
import torch

RAW = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/edit/"
FILES = ("manifest.json", "edit.npz")
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def fetch(cache_dir: str) -> tuple[dict, np.lib.npyio.NpzFile]:
    os.makedirs(cache_dir, exist_ok=True)
    for f in FILES:
        path = os.path.join(cache_dir, f)
        if not os.path.exists(path):
            urllib.request.urlretrieve(RAW + f, path)
    manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
    return manifest, np.load(os.path.join(cache_dir, "edit.npz"))


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def bits_to_bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)


def sha256(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


def build(
    originals: dict[str, torch.Tensor], cache_dir: str = ".cache/exp8_edit", verify: bool = True
) -> dict[str, torch.Tensor]:
    """originals: {tensor name: bf16-exact tensor} for the two edited tensors. Returns {name: edited float32}."""
    manifest, z = fetch(cache_dir)
    edits: dict[str, torch.Tensor] = {}
    for i, t in enumerate(manifest["tensors"]):
        name = t["name"]
        base = bf16_bits(originals[name])
        if list(base.shape) != t["shape"]:
            raise ValueError(f"{name}: shape {base.shape} != {t['shape']}")
        if verify and sha256(base) != t["base_sha256"]:
            raise ValueError(f"{name}: base tensor differs from revision {BASE_REVISION}")
        flat = base.copy().reshape(-1)
        flat[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        edited = flat.reshape(base.shape)
        if verify and sha256(edited) != t["edited_sha256"]:
            raise ValueError(f"{name}: rebuilt tensor does not match the evaluated checkpoint")
        edits[name] = bits_to_bf16(edited).float()
    return edits


def as_patch_script() -> str:
    """The same edit expressed as a standalone sandbox patch, for use as a GEPA seed or reference candidate."""
    return (
        "# Published Experiment 8 edit: sparse override of two down_proj tensors (layers 15 and 19)\n"
        "from dspy_weightspace.published_edit import build\n"
        "names = ['model.language_model.layers.15.mlp.down_proj.weight',\n"
        "         'model.language_model.layers.19.mlp.down_proj.weight']\n"
        "for n, w in build({n: load_original(n) for n in names}).items():\n"
        "    save_edit(n, w)\n"
    )
