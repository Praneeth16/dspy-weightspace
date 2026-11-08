"""AI Runtime smoke for the vLLM generation backend, before any long run relies on it.

1. Greedy generations after apply_patch(published) must match a separate vLLM engine loaded from the published
   checkpoint, and after reset must match base again: proof that in-place weight copies reach the model.
2. Times 32 MATH-train problems at MAX_NEW_TOKENS with the eval's sampling, to size the Track A budget.
"""

from __future__ import annotations

import importlib.util
import os
import time

import torch
from huggingface_hub import snapshot_download

from dspy_weightspace import HFLocalLM, load_math
from dspy_weightspace.data import to_chat
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.vllm_backend import VLLMBackend

E = os.environ
VOL, MAX_NEW = E["WS_VOLUME"], int(E.get("MAX_NEW_TOKENS", 16384))
here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else E.get("CODE_SOURCE_PATH", ".")
spec = importlib.util.spec_from_file_location("run_air_vllm", os.path.join(here, "run_air_vllm.py"))
rav = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rav)

base_dir = snapshot_download("Qwen/Qwen3.5-0.8B", revision=BASE_REVISION)
backend = VLLMBackend(base_dir, max_model_len=MAX_NEW + 2048, gpu_memory_utilization=0.35)
lm = HFLocalLM(revision=BASE_REVISION, gen_backend=backend)
dev, _ = load_math(n_dev=32, n_holdout=0)
chats = [to_chat(e.problem) for e in dev[:8]]

base0 = [c.text for c in lm.generate_detailed(chats, max_new_tokens=256, temperature=0.0)]
names = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in (15, 19)]
published = build({n: lm.original(n) for n in names}, cache_dir=f"{VOL}/exp8_edit")
lm.apply_patch(published)
patched = [c.text for c in lm.generate_detailed(chats, max_new_tokens=256, temperature=0.0)]
lm.reset()
base1 = [c.text for c in lm.generate_detailed(chats, max_new_tokens=256, temperature=0.0)]

ckpt, changed = rav.checkpoint(base_dir, "published", published)
from vllm import LLM, SamplingParams  # noqa: E402

ref = LLM(model=ckpt, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.25, enable_prefix_caching=False)
ref_out = [
    o.outputs[0].text
    for o in ref.generate(
        [lm._render(c) for c in chats], SamplingParams(temperature=0.0, max_tokens=256), use_tqdm=False
    )
]

same_ref = sum(a == b for a, b in zip(patched, ref_out, strict=True))
same_base = sum(a == b for a, b in zip(base0, base1, strict=True))
differs = sum(a != b for a, b in zip(base0, patched, strict=True))
print(
    f"patched == fresh published checkpoint: {same_ref}/8; base after reset == base: {same_base}/8; "
    f"patched differs from base: {differs}/8; checkpoint changed {changed:,}",
    flush=True,
)

t0 = time.time()
outs = lm.generate_detailed([to_chat(e.problem) for e in dev], max_new_tokens=MAX_NEW, seed=0)
dt = time.time() - t0
cut = sum(o.finish_reason == "length" for o in outs) / len(outs)
print(
    f"32 MATH-train problems at {MAX_NEW} tokens (base): {dt:.0f}s, cut {cut:.0%}, "
    f"{sum(o.n_tokens for o in outs) / dt:.0f} tok/s",
    flush=True,
)
ok = same_ref >= 7 and same_base == 8 and differs >= 1
print("SMOKE", "PASS" if ok else "FAIL", flush=True)
if not ok:
    raise SystemExit(1)
torch.cuda.empty_cache()
