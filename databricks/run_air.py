"""End-to-end run on Databricks AI Runtime (serverless GPU). Script twin of notebooks/databricks.py.

Configured by environment variables (set in workload.yaml): WS_VOLUME, REFLECTION_ENDPOINT, N_DEV, N_HOLDOUT,
MAX_NEW_TOKENS, PASSES_PER_ROUND, MAX_METRIC_CALLS, MAX_TURNS, MLFLOW_EXPERIMENT. Writes results.md, patch.py and
summary.json under $WS_VOLUME/runs/<timestamp>/ and logs the run to MLflow.
"""

from __future__ import annotations

import json
import os
import time

import mlflow
import torch

from dspy_weightspace import HFLocalLM, MathSolver, WeightSpaceOptimizer, load_gsm8k, math_metric
from dspy_weightspace.baseline import results_table, score_rows
from dspy_weightspace.published_edit import BASE_REVISION, as_patch_script, build
from dspy_weightspace.reflection import databricks_lm

E = os.environ
VOL = E["WS_VOLUME"]
ENDPOINT = E.get("REFLECTION_ENDPOINT", "databricks-claude-sonnet-5-5")
N_DEV, N_HOLD = int(E.get("N_DEV", 32)), int(E.get("N_HOLDOUT", 48))
MAX_NEW, PASSES = int(E.get("MAX_NEW_TOKENS", 4096)), int(E.get("PASSES_PER_ROUND", 6))
METRIC_CALLS, MAX_TURNS = int(E.get("MAX_METRIC_CALLS", 100)), int(E.get("MAX_TURNS", 30))
SEED_PATCH = as_patch_script() if E.get("SEED_PATCH", "") == "published" else ""
RUN_DIR = f"{VOL}/runs/{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=True)
print(
    {
        "endpoint": ENDPOINT,
        "n_dev": N_DEV,
        "n_holdout": N_HOLD,
        "max_new_tokens": MAX_NEW,
        "passes": PASSES,
        "metric_calls": METRIC_CALLS,
        "max_turns": MAX_TURNS,
        "run_dir": RUN_DIR,
        "cuda": torch.cuda.is_available(),
    },
    flush=True,
)

mlflow.set_tracking_uri("databricks")
# AI Runtime creates the MLflow run and exports MLFLOW_RUN_ID; start_run() below attaches to it.

lm = HFLocalLM(revision=BASE_REVISION, batch_size=16)
prog = MathSolver(lm, max_new_tokens=MAX_NEW)
dev, hold = load_gsm8k(n_dev=N_DEV, n_holdout=N_HOLD)
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
    log_dir=f"{RUN_DIR}/optimizer",
)
with mlflow.start_run() as run:
    mlflow.log_params(
        {
            "model": lm.model_name,
            "reflection": ENDPOINT,
            "n_dev": N_DEV,
            "n_holdout": N_HOLD,
            "max_new_tokens": MAX_NEW,
            "passes_per_round": PASSES,
            "max_metric_calls": METRIC_CALLS,
            "max_turns": MAX_TURNS,
            "seed_patch": "published" if SEED_PATCH else "none",
        }
    )
    t0 = time.time()
    optimized = opt.compile(prog, trainset=dev, valset=dev)
    wall = time.time() - t0
    rows["optimizer-found edit"] = score_rows(optimized, hold, math_metric)
    lm.reset()
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
