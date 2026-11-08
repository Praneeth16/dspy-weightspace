"""Direction x scope x strength sweep on layers 15 and 19, written as run dirs run_air_vllm.py can replay.

Every variant has the published edit's form, W[:, C] -= alpha * u (u^T W[:, C]) on down_proj, and changes one
thing: the direction u (published, difference-in-means, the cold optimizer run's, random), the columns C (the
published edit's 64 per layer, or all of them) or alpha. A sanity variant rebuilds the published edit from its own
direction and is compared to the published tensors, not evaluated.

    python scripts/make_sweep.py --profile fe-vm-lakebase-praneeth [--upload]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time

import numpy as np
import torch

from dspy_weightspace import HFLocalLM
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.sandbox import PatchSandbox, run_fresh

VOL = "dbfs:/Volumes/serverless_lakebase_praneeth_catalog/weightspace/artifacts"
DIFFMEAN = "runs/20261108-124746-diffmean"
COLD = "runs/20261104-063220/optimizer/artifacts"
LAYERS = (15, 19)

# label: (direction, scope, alpha)
VARIANTS = {
    "sanity-pub-C64-6": ("pub", "C64", 6.0),
    "pub-all-6": ("pub", "all", 6.0),
    "dm-C64-6": ("dm", "C64", 6.0),
    "dm-C64-9": ("dm", "C64", 9.0),
    "dm-all-9": ("dm", "all", 9.0),
    "cold-C64-6": ("cold", "C64", 6.0),
    "rand-C64-6": ("rand", "C64", 6.0),
}

PATCH = """# {label}: W[:, C] -= alpha * u (u^T W[:, C]) on down_proj, direction {direction}, columns {scope}
alpha = {alpha!r}
for L in {layers!r}:
    n = f'model.layers.{{L}}.mlp.down_proj.weight'
    W = load_original(n)
    u = load_artifact(f'u_L{{L + 1}}').float().flatten()
    u = u / u.norm()
    C = {cols}
    W[:, C] = W[:, C] - alpha * torch.outer(u, u @ W[:, C])
    save_edit(n, W)
"""


def fetch(profile: str, src: str, dest: str) -> None:
    cmd = ["databricks", "fs", "cp", "-r", f"{VOL}/{src}", dest, "--overwrite", "--profile", profile]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    lm = HFLocalLM(revision=BASE_REVISION, device="cpu", dtype=torch.bfloat16)

    fetch(a.profile, "exp8_edit", f"{tmp}/exp8_edit")
    names = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in LAYERS]
    published = {lm.resolve(n): t for n, t in build({n: lm.original(n) for n in names}, f"{tmp}/exp8_edit").items()}

    dirs: dict[str, dict[int, np.ndarray]] = {"pub": {}, "dm": {}, "cold": {}, "rand": {}}
    cols: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for L in LAYERS:
        name = lm.resolve(f"model.layers.{L}.mlp.down_proj.weight")
        d = (published[name] - lm.original(name)).double()
        c = (d != 0).any(0).nonzero().ravel()
        cols[L] = c.numpy().astype(np.float32)
        dirs["pub"][L + 1] = torch.linalg.svd(d[:, c], full_matrices=False).U[:, 0].float().numpy()
        dirs["rand"][L + 1] = rng.standard_normal(d.shape[0]).astype(np.float32)
    fetch(a.profile, DIFFMEAN, f"{tmp}/dm")
    fetch(a.profile, COLD, f"{tmp}/cold")
    for L in LAYERS:
        dirs["dm"][L + 1] = np.load(f"{tmp}/dm/dirs_L{L + 1}.npy")
        dirs["cold"][L + 1] = np.load(f"{tmp}/cold/loopdir_L{L + 1}.npy").ravel()

    stamp = f"runs/{time.strftime('%Y%m%d-%H%M%S')}-sweep"
    root = f"{tmp}/{stamp}"
    print(f"building {len(VARIANTS)} variants in {root}")
    for label, (direction, scope, alpha) in VARIANTS.items():
        work = f"{root}/{label}/optimizer"
        sb = PatchSandbox(lm, work)
        for L in LAYERS:
            sb.save_artifact(f"u_L{L + 1}", torch.from_numpy(dirs[direction][L + 1]))
            sb.save_artifact(f"cols_L{L}", torch.from_numpy(cols[L]))
        c_expr = "load_artifact(f'cols_L{L}').long().flatten()" if scope == "C64" else "slice(None)"
        src = PATCH.format(label=label, direction=direction, scope=scope, alpha=alpha, layers=list(LAYERS), cols=c_expr)
        with open(f"{root}/{label}/patch.py", "w") as f:
            f.write(src)
        fresh, res = run_fresh(lm, src, work)
        assert res.ok, (label, res.error)
        fresh.apply()
        changed = lm.changed_entries()
        lm.reset()
        note = ""
        if label.startswith("sanity"):
            rel = max(
                float((fresh.edits[n] - published[n]).norm() / (published[n] - lm.original(n)).norm())
                for n in published
            )
            note = f"  relative difference from the published delta: {rel:.4f}"
        print(f"  {label}: {changed:,} weights changed{note}")

    if a.upload:
        cmd = ["databricks", "fs", "cp", "-r", root, f"{VOL}/{stamp}", "--overwrite", "--profile", a.profile]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"uploaded to {VOL}/{stamp}")
    print("local:", root)


if __name__ == "__main__":
    main()
