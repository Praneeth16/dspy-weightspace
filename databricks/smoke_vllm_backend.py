"""AI Runtime smoke for the vLLM generation backend, before any long run relies on it.

1. Reference greedy outputs for base and for the published checkpoint come from separate processes, each a fresh
   vLLM engine with the same settings as the backend. The backend then generates base, applies the published edit
   in place, generates, resets, generates. If in-place copies reach the model, backend-vs-reference agreement on the
   patched model should match the agreement on base (engines can differ numerically; that sets the bar).
2. Times 32 MATH-train problems at MAX_NEW_TOKENS with the eval's sampling, to size the Track A budget.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

from huggingface_hub import snapshot_download

from dspy_weightspace import HFLocalLM, load_math
from dspy_weightspace.data import to_chat
from dspy_weightspace.published_edit import BASE_REVISION, build

E = os.environ
VOL, MAX_NEW = E["WS_VOLUME"], int(E.get("MAX_NEW_TOKENS", 16384))
ENGINE = {
    "dtype": "bfloat16",
    "max_model_len": MAX_NEW + 2048,
    "gpu_memory_utilization": 0.5,
    "enable_prefix_caching": False,
    "seed": 0,
}
REF = """
import json, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from dspy_weightspace.vllm_backend import drop_stale_flash_attn
drop_stale_flash_attn()
from vllm import LLM, SamplingParams
model, prompts_file, out_file, engine = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
prompts = json.load(open(prompts_file))
outs = LLM(model=model, **engine).generate(prompts, SamplingParams(temperature=0.0, max_tokens=256), use_tqdm=False)
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
names = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in (15, 19)]
published = build({n: cpu_lm.original(n) for n in names}, cache_dir=f"{VOL}/exp8_edit")
ckpt, changed = rav.checkpoint(base_dir, "published", published)
del cpu_lm

json.dump(prompts, open("/tmp/smoke_prompts.json", "w"))
refs = {}
for label, model in (("base", base_dir), ("published", ckpt)):
    subprocess.run(
        [sys.executable, "-c", REF, model, "/tmp/smoke_prompts.json", f"/tmp/ref_{label}.json", json.dumps(ENGINE)],
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
ok = base1 == base0 and pp >= pb - 0.15 and pp > px
print(
    "SMOKE", "PASS" if ok else "FAIL", f"(prefix agreement: base {pb:.0%}, patched {pp:.0%}, patched-vs-base {px:.0%})"
)
if not ok:
    raise SystemExit(1)
