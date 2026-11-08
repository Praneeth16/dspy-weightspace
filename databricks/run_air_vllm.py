"""Full-protocol vLLM evaluation on AI Runtime: base, the published Experiment-8 edit, and edits the optimizer found.

Each edited variant is a copy of the base checkpoint with only the edited tensors replaced, so vLLM loads the same
architecture and config as for the base model. Each variant runs in its own process because vLLM holds GPU memory
until exit.

Env: WS_VOLUME, DATASET, N, SAMPLES, MAX_TOKENS, CHUNK, RUN_DIR, PROTOCOL, PRESENCE_PENALTY, VARIANTS = comma list of
  base | published | <label>:<run dir under WS_VOLUME>   (the run dir's patch.py, replayed on its optimizer artifacts)
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

import mlflow
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from dspy_weightspace import HFLocalLM
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.sandbox import PatchSandbox

E = os.environ
VOL = E["WS_VOLUME"]
VARIANTS = [v.strip() for v in E.get("VARIANTS", "base,published").split(",") if v.strip()]
DATASET = E.get("DATASET", "HuggingFaceH4/MATH-500")
N, SAMPLES = int(E.get("N", 500)), int(E.get("SAMPLES", 2))
MAX_TOKENS = E.get("MAX_TOKENS", "")  # empty: the protocol's own budget
CHUNK = int(E.get("CHUNK", 100))
PROTOCOL, PRESENCE = E.get("PROTOCOL", "think35"), float(E.get("PRESENCE_PENALTY", 1.5))
PUBLISHED = [
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
]


def edits_for(variant: str, lm: HFLocalLM) -> dict[str, torch.Tensor]:
    if variant == "base":
        return {}
    if variant == "published":
        return build({n: lm.original(n) for n in PUBLISHED}, cache_dir=f"{VOL}/exp8_edit")
    run = f"{VOL}/{variant.split(':', 1)[1]}"
    sb = PatchSandbox(lm, f"{run}/optimizer")
    with open(f"{run}/patch.py") as f:
        res = sb.run(f.read())
    if not res.ok:
        raise RuntimeError(f"{variant}: patch failed\n{res.error}")
    return sb.edits


def checkpoint(base_dir: str, label: str, edits: dict[str, torch.Tensor]) -> tuple[str, int]:
    """Write base_dir with the edited tensors swapped in; returns (path, number of changed entries)."""
    if not edits:
        return base_dir, 0
    out = f"/tmp/ckpt/{label}"
    shutil.copytree(base_dir, out, dirs_exist_ok=True)  # follows the HF cache symlinks, so these are real files
    pending = {n.replace("model.language_model.", "model.", 1): t for n, t in edits.items()}
    changed = 0
    for shard in sorted(glob.glob(f"{out}/*.safetensors")):
        with safe_open(shard, "pt") as f:
            keys, meta = set(f.keys()), f.metadata()
        hits = {}
        for short in list(pending):
            for key in (short, short.replace("model.", "model.language_model.", 1)):
                if key in keys:
                    hits[key] = pending.pop(short)
                    break
        if not hits:
            continue
        tensors = load_file(shard)
        for key, t in hits.items():
            new = t.to(tensors[key].dtype)
            changed += int((new != tensors[key]).sum())
            tensors[key] = new
        save_file(tensors, f"{shard}.tmp", metadata=meta)
        os.replace(f"{shard}.tmp", shard)
    if pending:
        raise KeyError(f"{label}: tensors not found in checkpoint: {sorted(pending)}")
    return out, changed


def drop_stale_flash_attn() -> None:
    """The image's flash-attn is compiled against its own torch; a newer vLLM brings a newer torch, and vLLM then
    imports the stale extension (undefined symbol) for the vision tower's rotary embedding. vLLM ships its own
    attention kernels, so the package is removed instead of rebuilt."""
    from importlib import metadata

    try:
        dist = metadata.distribution("flash-attn")
    except metadata.PackageNotFoundError:
        return
    version, root = dist.version, dist.locate_file("")
    files = [root / f for f in dist.files or []]
    for path in files:
        if path.is_file():
            path.unlink()
    for name in ("flash_attn", "hopper"):
        shutil.rmtree(root / name, ignore_errors=True)
    for d in sorted({p.parent for p in files}, key=lambda d: len(d.parts), reverse=True):
        if d.is_dir() and d != root and not any(d.iterdir()):
            d.rmdir()  # an empty .dist-info would still register as an installed distribution
    print(f"removed stale flash-attn {version}", flush=True)


def evaluate(path: str, out: str) -> dict:
    cmd = [sys.executable, "-m", "dspy_weightspace.vllm_eval", "--model", path, "--dataset", DATASET]
    cmd += ["--n", str(N), "--samples", str(SAMPLES), "--out", out] + (
        ["--max-tokens", MAX_TOKENS] if MAX_TOKENS else []
    )
    cmd += ["--chunk", str(CHUNK), "--protocol", PROTOCOL, "--presence-penalty", str(PRESENCE)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    result = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT ") :])
    if proc.wait() != 0 or result is None:
        raise RuntimeError(f"vllm_eval exited {proc.returncode} for {path}")
    return result


def table(rows: dict[str, dict]) -> str:
    lines = [
        "| Variant | Accuracy | Cut at limit | Mean tokens | Generations | Weights changed |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in rows.items():
        lines.append(
            f"| {label} | {r['accuracy']:.1%} | {r['cut_at_limit']:.1%} | {r['mean_tokens']:.0f} | "
            f"{r['generations']} | {r['weights_changed']:,} |"
        )
    return "\n".join(lines)


def main() -> None:
    # RUN_DIR (under WS_VOLUME) reuses an earlier run's directory: finished problems in its .jsonl files are kept.
    run_dir = f"{VOL}/{E['RUN_DIR']}" if E.get("RUN_DIR") else f"{VOL}/runs/{time.strftime('%Y%m%d-%H%M%S')}-vllm"
    os.makedirs(run_dir, exist_ok=True)
    print(
        {
            "variants": VARIANTS,
            "dataset": DATASET,
            "n": N,
            "samples": SAMPLES,
            "max_tokens": MAX_TOKENS or "protocol",
            "protocol": PROTOCOL,
            "presence_penalty": PRESENCE,
            "run_dir": run_dir,
        }
    )
    drop_stale_flash_attn()
    base_dir = snapshot_download("Qwen/Qwen3.5-0.8B", revision=BASE_REVISION)
    lm = HFLocalLM(revision=BASE_REVISION, device="cpu", dtype=torch.bfloat16)

    rows: dict[str, dict] = {}
    with mlflow.start_run():
        mlflow.log_params(
            {
                "variants": ",".join(VARIANTS),
                "dataset": DATASET,
                "n": N,
                "samples": SAMPLES,
                "max_tokens": MAX_TOKENS or "protocol",
                "protocol": PROTOCOL,
                "presence_penalty": PRESENCE,
            }
        )
        for i, variant in enumerate(VARIANTS):
            label = variant.split(":", 1)[0] + (f"-{PROTOCOL}" if PROTOCOL != "think35" else "")
            label += f"-pp{PRESENCE:g}" if PRESENCE != 1.5 else ""
            t0 = time.time()
            try:
                path, changed = checkpoint(base_dir, label, edits_for(variant, lm))
                print(f"{label}: {changed:,} weights changed, checkpoint {path}", flush=True)
                rows[label] = {**evaluate(path, f"{run_dir}/{label}.jsonl"), "weights_changed": changed}
            except Exception as e:  # noqa: BLE001 - one broken variant must not cost the others
                print(f"{label} FAILED: {e}", flush=True)
                continue
            rows[label]["seconds"] = time.time() - t0
            mlflow.log_metrics(
                {f"{i}_{label}_{m}": rows[label][m] for m in ("accuracy", "cut_at_limit", "mean_tokens")}
            )
            with open(f"{run_dir}/summary.json", "w") as f:
                json.dump(rows, f, indent=2)
            print(table(rows), flush=True)
        mlflow.log_text(table(rows), "results.md")
    with open(f"{run_dir}/results.md", "w") as f:
        f.write(table(rows) + "\n")
    print("SUMMARY " + json.dumps(rows), flush=True)
    failed = [v for v in VARIANTS if v.split(":", 1)[0] not in rows]
    if failed:
        raise RuntimeError(f"variants failed: {failed}")


if __name__ == "__main__":
    main()
