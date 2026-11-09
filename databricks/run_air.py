"""End-to-end run on Databricks AI Runtime (serverless GPU). Script twin of notebooks/databricks.py.

Configured by environment variables (set in workload.yaml): WS_VOLUME, REFLECTION_ENDPOINT, N_DEV, N_FEEDBACK,
N_HOLDOUT, MAX_NEW_TOKENS, PASSES_PER_ROUND, MAX_METRIC_CALLS, MAX_TURNS, SCORE_CANDIDATES, DATA, GEN_BACKEND,
VLLM_GPU_UTIL, SEED. Writes results.md, patch.py, candidate_curve.json and summary.json under
$WS_VOLUME/runs/<timestamp>/ and logs the run to MLflow.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import mlflow
import torch

from dspy_weightspace import HFLocalLM, MathSolver, WeightSpaceOptimizer, load_gsm8k, load_math, math_metric
from dspy_weightspace.adapter import COMPONENT
from dspy_weightspace.baseline import results_table, score_rows
from dspy_weightspace.published_edit import BASE_REVISION, as_patch_script, build
from dspy_weightspace.reflection import databricks_lm
from dspy_weightspace.sandbox import run_fresh

E = os.environ
VOL = E["WS_VOLUME"]
ENDPOINT = E.get("REFLECTION_ENDPOINT", "databricks-claude-sonnet-5-5")
N_DEV, N_HOLD = int(E.get("N_DEV", 48)), int(E.get("N_HOLDOUT", 48))
# Dev splits into feedback (probes, reflection minibatches) and selection (GEPA valset); the holdout is never seen.
N_FEEDBACK = int(E.get("N_FEEDBACK", 16))
SCORE_CANDIDATES = E.get("SCORE_CANDIDATES", "1") != "0"
# DATA=math selects on MATH-train (closer to the MATH-500 test than GSM8K); GEN_BACKEND=vllm generates with vLLM
# (with the eval's sampling, presence penalty included) so long traces are affordable.
DATA, GEN_BACKEND = E.get("DATA", "gsm8k"), E.get("GEN_BACKEND", "hf")
MAX_NEW, PASSES = int(E.get("MAX_NEW_TOKENS", 4096)), int(E.get("PASSES_PER_ROUND", 6))
METRIC_CALLS, MAX_TURNS = int(E.get("MAX_METRIC_CALLS", 100)), int(E.get("MAX_TURNS", 30))
# SEED varies only the search (GEPA sampling); the data splits stay fixed so repeat runs are comparable.
SEED = int(E.get("SEED", 0))
SEED_PATCH = as_patch_script() if E.get("SEED_PATCH", "") == "published" else ""
# Timestamp plus a random suffix: two jobs started in the same second once shared a dir and clobbered each other.
RUN_DIR = f"{VOL}/runs/{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
os.makedirs(RUN_DIR, exist_ok=True)
print(
    {
        "endpoint": ENDPOINT,
        "n_dev": N_DEV,
        "n_holdout": N_HOLD,
        "max_new_tokens": MAX_NEW,
        "data": DATA,
        "gen_backend": GEN_BACKEND,
        "passes": PASSES,
        "metric_calls": METRIC_CALLS,
        "max_turns": MAX_TURNS,
        "seed": SEED,
        "run_dir": RUN_DIR,
        "cuda": torch.cuda.is_available(),
    },
    flush=True,
)

mlflow.set_tracking_uri("databricks")
# AI Runtime creates the MLflow run and exports MLFLOW_RUN_ID; start_run() below attaches to it.

backend = None
if GEN_BACKEND == "vllm":
    from huggingface_hub import snapshot_download

    from dspy_weightspace.vllm_backend import VLLMBackend

    model_dir = snapshot_download("Qwen/Qwen3.5-0.8B", revision=BASE_REVISION)
    backend = VLLMBackend(
        model_dir, max_model_len=MAX_NEW + 2048, gpu_memory_utilization=float(E.get("VLLM_GPU_UTIL", 0.5))
    )
lm = HFLocalLM(revision=BASE_REVISION, batch_size=16, gen_backend=backend)
prog = MathSolver(lm, max_new_tokens=MAX_NEW)
dev, hold = (load_math if DATA == "math" else load_gsm8k)(n_dev=N_DEV, n_holdout=N_HOLD)
feedback, select = dev[:N_FEEDBACK], dev[N_FEEDBACK:]
print(lm.device, lm.dtype, type(lm.hf).__name__, len(lm.names), "tensors", flush=True)

rows = {}
t0 = time.time()
rows["base, T=0.6 (blog sampling)"] = score_rows(prog, hold, math_metric)
print(f"{time.time() - t0:.0f}s per {N_HOLD}-problem pass", flush=True)
rows["base, T=1.0 (Qwen card)"] = score_rows(prog, hold, math_metric, temperature=1.0)

NAMES = ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"]
published = build({n: lm.original(n) for n in NAMES}, cache_dir=f"{VOL}/exp8_edit")
lm.apply_patch(published)
rows["published Exp-8 edit (131k weights)"] = score_rows(prog, hold, math_metric)
lm.reset()
# Dose-response on the published delta, the direction the warm-start round explored: W = W0 + k * (W_pub - W0).
for k in [float(x) for x in E.get("EXTRA_SCALES", "").split(",") if x.strip()]:
    lm.apply_patch({n: lm.original(n) + k * (published[n] - lm.original(n)) for n in NAMES})
    rows[f"published delta x {k:g}"] = score_rows(prog, hold, math_metric)
    lm.reset()
print(results_table(rows), flush=True)

reflection = databricks_lm(ENDPOINT)
opt = WeightSpaceOptimizer(
    math_metric,
    reflection,
    max_metric_calls=METRIC_CALLS,
    passes_per_round=PASSES,
    max_items_per_pass=24,
    reflection_minibatch_size=8,
    max_turns=MAX_TURNS,
    seed_patch=SEED_PATCH,
    seed=SEED,
    log_dir=f"{RUN_DIR}/optimizer",
)
with mlflow.start_run() as run:
    mlflow.log_params(
        {
            "model": lm.model_name,
            "reflection": ENDPOINT,
            "n_dev": N_DEV,
            "n_feedback": N_FEEDBACK,
            "n_holdout": N_HOLD,
            "max_new_tokens": MAX_NEW,
            "data": DATA,
            "gen_backend": GEN_BACKEND,
            "passes_per_round": PASSES,
            "max_metric_calls": METRIC_CALLS,
            "max_turns": MAX_TURNS,
            "seed": SEED,
            "seed_patch": "published" if SEED_PATCH else "none",
        }
    )
    t0 = time.time()
    optimized = opt.compile(prog, trainset=feedback, valset=select)
    wall = time.time() - t0
    rows["optimizer-found edit"] = score_rows(optimized, hold, math_metric)
    lm.reset()
    # Retrospective holdout score of every candidate, after selection is final: shows whether selection-set gains
    # carried to new problems. These scores never feed back into the search.
    curve = []
    if SCORE_CANDIDATES:
        res = optimized.detailed_results
        for i, cand in enumerate(res.candidates):
            src = cand[COMPONENT]
            sb, out = run_fresh(lm, src, f"{RUN_DIR}/optimizer") if src.strip() else (None, None)
            if out is not None and not out.ok:
                curve.append({"candidate": i, "error": out.error[:300]})
                continue
            if sb is not None:
                sb.apply()
            r = score_rows(prog, hold, math_metric)
            lm.reset()
            curve.append({"candidate": i, "parents": res.parents[i], "selection": res.val_aggregate_scores[i], **r})
            print("candidate", curve[-1], flush=True)
        with open(f"{RUN_DIR}/candidate_curve.json", "w") as f:
            json.dump(curve, f, indent=2)
        mlflow.log_text(json.dumps(curve, indent=2), "candidate_curve.json")
    table = results_table(rows)
    mlflow.log_text(optimized.patch_src or "# no edit", "patch.py")
    mlflow.log_text(table, "results.md")
    mlflow.log_metrics(
        {
            "weights_changed": optimized.weights_changed,
            "wall_seconds": wall,
            "gepa_metric_calls": optimized.detailed_results.total_metric_calls or 0,
            **{
                f"holdout_{i}_{m}": v[m]
                for i, v in enumerate(rows.values())
                for m in ("accuracy", "cut_rate", "loop_rate")
            },
        }
    )

print(table, flush=True)
print(
    "edited tensors:",
    optimized.edited_tensors,
    "weights changed:",
    optimized.weights_changed,
    f"wall {wall / 60:.0f} min",
)
print("PATCH:\n" + (optimized.patch_src or "# no edit"), flush=True)
out = {
    "rows": rows,
    "weights_changed": optimized.weights_changed,
    "edited_tensors": optimized.edited_tensors,
    "wall_seconds": wall,
    "gepa_candidates": len(optimized.detailed_results.candidates),
    "candidate_curve": curve,
    "run_dir": RUN_DIR,
    "mlflow_run_id": run.info.run_id,
}
with open(f"{RUN_DIR}/summary.json", "w") as f:
    json.dump(out, f, indent=2)
with open(f"{RUN_DIR}/results.md", "w") as f:
    f.write(table + "\n")
with open(f"{RUN_DIR}/patch.py", "w") as f:
    f.write(optimized.patch_src or "# no edit\n")
print("SUMMARY " + json.dumps(out), flush=True)
