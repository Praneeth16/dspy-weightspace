"""Find MATH-train problems the base model already solves with thinking on: the guard set.

An edit that makes the model stop sooner could rush problems it used to solve. Searches score every candidate on
this set as well (GUARD_SET in run_air.py). The problems are disjoint from the feedback, selection and held-out sets.
Each problem is sampled twice; it enters the set if either sample is correct. Writes $WS_VOLUME/guard/guard_solved.json.
"""

from __future__ import annotations

import json
import os

from huggingface_hub import snapshot_download

from dspy_weightspace import HFLocalLM, MathSolver, load_math, math_metric
from dspy_weightspace.published_edit import BASE_REVISION
from dspy_weightspace.vllm_backend import VLLMBackend

E = os.environ
VOL, MAX_NEW = E["WS_VOLUME"], int(E.get("MAX_NEW_TOKENS", 16384))
N_POOL, SAMPLES, SEED = int(E.get("N_POOL", 500)), int(E.get("SAMPLES", 2)), int(E.get("GUARD_SEED", 11))

model_dir = snapshot_download("Qwen/Qwen3.5-0.8B", revision=BASE_REVISION)
backend = VLLMBackend(
    model_dir, max_model_len=MAX_NEW + 2048, gpu_memory_utilization=float(E.get("VLLM_GPU_UTIL", 0.5))
)
lm = HFLocalLM(revision=BASE_REVISION, gen_backend=backend)
prog = MathSolver(lm, max_new_tokens=MAX_NEW)

dev, hold = load_math(n_dev=48, n_holdout=48)  # the sets every search uses
used = {e.problem for e in dev + hold}
pool, _ = load_math(n_dev=N_POOL + len(used), n_holdout=0, seed=SEED)
pool = [e for e in pool if e.problem not in used][:N_POOL]

hits = [0] * len(pool)
tokens = [[] for _ in pool]
for s in range(SAMPLES):
    preds = prog.batch_forward([e.problem for e in pool], seed=s)
    for i, (e, p) in enumerate(zip(pool, preds, strict=True)):
        if math_metric(e, p).correct:
            hits[i] += 1
            tokens[i].append(p.n_tokens)
    print(f"sample {s}: {sum(h > 0 for h in hits)} of {len(pool)} solved so far", flush=True)

solved = [
    {"problem": e.problem, "answer": e.answer, "base_correct": h, "base_tokens": t}
    for e, h, t in zip(pool, hits, tokens, strict=True)
    if h
]
if not solved:
    raise RuntimeError("the base model solved no problem in the pool")
os.makedirs(f"{VOL}/guard", exist_ok=True)
with open(f"{VOL}/guard/guard_solved.json", "w") as f:
    json.dump(solved, f, indent=1)
print(
    f"GUARD {len(solved)} problems, solved in both samples: {sum(g['base_correct'] == SAMPLES for g in solved)}",
    flush=True,
)
