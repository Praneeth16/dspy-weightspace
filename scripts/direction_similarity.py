"""How similar are the directions behind the edits? Cold and warm optimizer directions, the published edit's own
direction, and (once run) the no-LLM difference-in-means direction, compared layer by layer.

Also reads the published edit as a projection: on its 64 columns per layer, how much of the change is rank one, and
what alpha fits dW = -alpha * u u^T W0. Run locally; it copies the small artifacts out of the UC volume.

    python scripts/direction_similarity.py --profile fe-vm-lakebase-praneeth [--diffmean runs/<stamp>-diffmean]
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from dspy_weightspace.published_edit import BASE_REVISION, build

VOL = "dbfs:/Volumes/serverless_lakebase_praneeth_catalog/weightspace/artifacts"
COLD, WARM = "runs/20261104-063220/optimizer/artifacts", "runs/20261104-063705/optimizer/artifacts"


def pull(profile: str, src: str, pattern: str, dest: str) -> dict[int, np.ndarray]:
    os.makedirs(dest, exist_ok=True)
    subprocess.run(
        ["databricks", "fs", "cp", "-r", f"{VOL}/{src}", dest, "--overwrite", "--profile", profile],
        check=True,
        capture_output=True,
    )
    out = {}
    for path in glob.glob(f"{dest}/{pattern}"):
        out[int(path.rsplit("_L", 1)[1].split(".")[0])] = np.load(path).astype(np.float64).ravel()
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--diffmean", default="", help="baseline run dir under the volume, e.g. runs/<stamp>-diffmean")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()

    sets = {
        "cold": pull(a.profile, COLD, "loopdir_L*.npy", f"{tmp}/cold"),
        "warm": pull(a.profile, WARM, "doubt_u_L*.npy", f"{tmp}/warm"),
    }
    if a.diffmean:
        sets["diffmean"] = pull(a.profile, a.diffmean, "dirs_L*.npy", f"{tmp}/diffmean")

    # Published edit: base tensors from the HF snapshot, edit rebuilt and hash-checked by build().
    names = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in (15, 19)]
    shard = glob.glob(f"{snapshot_download('Qwen/Qwen3.5-0.8B', revision=BASE_REVISION)}/*.safetensors")[0]
    with safe_open(shard, "pt") as f:
        base = {n: f.get_tensor(n).float() for n in names}
    subprocess.run(
        ["databricks", "fs", "cp", "-r", f"{VOL}/exp8_edit", f"{tmp}/exp8_edit", "--overwrite", "--profile", a.profile],
        check=True,
        capture_output=True,
    )
    edited = build(base, cache_dir=f"{tmp}/exp8_edit")
    pub = {}
    print("published edit, per layer (restricted to its changed columns):")
    for n in names:
        layer = int(n.split(".layers.")[1].split(".")[0])
        d = (edited[n] - base[n]).double()
        cols = (d != 0).any(0).nonzero().ravel()
        dc, w0 = d[:, cols], base[n].double()[:, cols]
        s = torch.linalg.svdvals(dc)
        u = torch.linalg.svd(dc, full_matrices=False).U[:, 0]
        proj = torch.outer(u, u @ w0)
        alpha = float(-(dc * proj).sum() / (proj * proj).sum())
        resid = float((dc + alpha * proj).norm() / dc.norm())
        pub[layer + 1] = u.numpy()
        print(
            f"  L{layer}: {len(cols)} columns, rank-1 share {float(s[0] ** 2 / (s**2).sum()):.3f}, "
            f"fit dW = -{alpha:.2f} u u^T W0 (relative residual {resid:.3f})"
        )
    sets["published"] = pub

    print("\n|cosine| between directions, by hidden-state index (down_proj of layer L writes index L+1):")
    keys = sorted(sets)
    for i, x in enumerate(keys):
        for y in keys[i + 1 :]:
            common = sorted(set(sets[x]) & set(sets[y]))
            if common:
                vals = ", ".join(f"h{k}: {cos(sets[x][k], sets[y][k]):.2f}" for k in common)
                print(f"  {x} vs {y}: {vals}")
    rng = np.random.default_rng(0)
    chance = np.mean([cos(*rng.standard_normal((2, 1024))) for _ in range(200)])
    print(f"  (two random directions in this space: |cos| ~ {chance:.2f})")


if __name__ == "__main__":
    main()
