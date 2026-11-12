"""AI Runtime smoke for the vLLM generation backend, before any long run relies on it.

1. Reference greedy outputs for base and for an edited checkpoint come from separate processes, each a fresh
   vLLM engine with the same settings as the backend. The backend then generates base, applies the same edit in
   place, generates, resets, generates. SMOKE_EDIT=published uses the published edit (down_proj only).
   SMOKE_EDIT=packed perturbs the projections vLLM stores stacked in one matrix (q/k/v, gate/up and the
   linear-attention input projections): a patch to gate_proj once failed to reach vLLM and hung a search.
   If in-place copies reach the model, backend-vs-reference agreement on the patched model should match the
   agreement on base (engines can differ numerically; that sets the bar).
2. Times 32 MATH-train problems at MAX_NEW_TOKENS with the eval's sampling, to size the Track A budget.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

import torch
from huggingface_hub import snapshot_download

from dspy_weightspace import HFLocalLM, load_math
from dspy_weightspace.data import to_chat
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.vllm_backend import THINK35

E = os.environ
VOL, MAX_NEW = E["WS_VOLUME"], int(E.get("MAX_NEW_TOKENS", 16384))
ENGINE = {
    "dtype": "bfloat16",
    "max_model_len": MAX_NEW + 2048,
    "gpu_memory_utilization": 0.5,
    "enable_prefix_caching": False,
    "seed": 0,
}
# Greedy, but with the backend's other sampling settings: presence_penalty and top_k change the argmax too.
GREEDY = {**THINK35, "temperature": 0.0}
REF = """
import json, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from dspy_weightspace.vllm_backend import drop_stale_flash_attn
drop_stale_flash_attn()
from vllm import LLM, SamplingParams
model, prompts_file, out_file, engine, sampling = sys.argv[1], sys.argv[2], sys.argv[3], *map(json.loads, sys.argv[4:6])
prompts = json.load(open(prompts_file))
outs = LLM(model=model, **engine).generate(prompts, SamplingParams(max_tokens=256, **sampling), use_tqdm=False)
json.dump([o.outputs[0].text for o in outs], open(out_file, "w"))
"""

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("run_air_vllm", os.path.join(here, "run_air_vllm.py"))
rav = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rav)


def agreement(a: list[str], b: list[str]) -> tuple[int, float]:
    """(exact matches, mean fraction of the shorter text that is a common prefix)."""
    frac = []
    for x, y in zip(a, b, strict=True):
        n = next((i for i, (p, q) in enumerate(zip(x, y, strict=False)) if p != q), min(len(x), len(y)))
        frac.append(n / max(1, min(len(x), len(y))))
    return sum(x == y for x, y in zip(a, b, strict=True)), sum(frac) / len(frac)


base_dir = snapshot_download("Qwen/Qwen3.5-0.8B", revision=BASE_REVISION)
cpu_lm = HFLocalLM(revision=BASE_REVISION, device="cpu")
dev, _ = load_math(n_dev=32, n_holdout=0)
prompts = [cpu_lm._render(to_chat(e.problem)) for e in dev[:8]]
EDIT = E.get("SMOKE_EDIT", "published")
if EDIT == "published":
    names = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in (15, 19)]
    published = build({n: cpu_lm.original(n) for n in names}, cache_dir=f"{VOL}/exp8_edit")
else:
    parts = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    # Layer 3 is a full-attention layer and layer 4 a linear-attention one, so every packed kind is covered.
    names = [
        n for n in cpu_lm.names if (".layers.3." in n or ".layers.4." in n) and any(f".{q}.weight" in n for q in parts)
    ]
    missing = [q for q in parts if not any(f".{q}.weight" in n for n in names)]
    if missing:
        raise RuntimeError(f"packed smoke: no tensor found for {missing}; names: {names}")
    g = torch.Generator().manual_seed(0)
    published = {n: cpu_lm.original(n) * (1 + 0.1 * torch.randn(cpu_lm.original(n).shape, generator=g)) for n in names}
    print("packed smoke edits:", [n.split("layers.")[1] for n in names], flush=True)
ckpt, changed = rav.checkpoint(base_dir, EDIT, published)
del cpu_lm

json.dump(prompts, open("/tmp/smoke_prompts.json", "w"))
refs = {}
for label, model in (("base", base_dir), ("published", ckpt)):
    subprocess.run(
        [
            sys.executable,
            "-c",
            REF,
            model,
            "/tmp/smoke_prompts.json",
            f"/tmp/ref_{label}.json",
            json.dumps(ENGINE),
            json.dumps(GREEDY),
        ],
        check=True,
    )
    refs[label] = json.load(open(f"/tmp/ref_{label}.json"))

from dspy_weightspace.vllm_backend import VLLMBackend  # noqa: E402  (after the reference processes released the GPU)

backend = VLLMBackend(base_dir, max_model_len=ENGINE["max_model_len"], gpu_memory_utilization=0.5)
lm = HFLocalLM(revision=BASE_REVISION, gen_backend=backend)
chats = [to_chat(e.problem) for e in dev[:8]]
gen = lambda: [c.text for c in lm.generate_detailed(chats, max_new_tokens=256, temperature=0.0)]  # noqa: E731
base0 = gen()
lm.apply_patch(published)
patched = gen()
lm.reset()
base1 = gen()

print(f"checkpoint changed {changed:,}", flush=True)
for name, a, b in (
    ("backend base vs fresh base", base0, refs["base"]),
    ("backend patched vs fresh published", patched, refs["published"]),
    ("backend patched vs fresh base", patched, refs["base"]),
    ("backend base after reset vs before", base1, base0),
):
    exact, prefix = agreement(a, b)
    print(f"  {name}: exact {exact}/8, mean common prefix {prefix:.0%}", flush=True)

t0 = time.time()
outs = lm.generate_detailed([to_chat(e.problem) for e in dev], max_new_tokens=MAX_NEW, seed=0)
dt = time.time() - t0
cut = sum(o.finish_reason == "length" for o in outs) / len(outs)
print(
    f"32 MATH-train problems at {MAX_NEW} tokens (base): {dt:.0f}s, cut {cut:.0%}, "
    f"{sum(o.n_tokens for o in outs) / dt:.0f} tok/s",
    flush=True,
)

pb, pp, px = (
    agreement(base0, refs["base"])[1],
    agreement(patched, refs["published"])[1],
    agreement(patched, refs["base"])[1],
)
# Same sampling everywhere, so agreement should be near-total; the patched model must sit with published, not base.
ok = base1 == base0 and pb >= 0.9 and pp >= 0.9 and px <= pp - 0.2
print(
    "SMOKE", "PASS" if ok else "FAIL", f"(prefix agreement: base {pb:.0%}, patched {pp:.0%}, patched-vs-base {px:.0%})"
)
if not ok:
    raise SystemExit(1)
