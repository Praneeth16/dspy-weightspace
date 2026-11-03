# Databricks notebook source
# MAGIC %md
# MAGIC # Weight-space programming as a DSPy optimizer
# MAGIC
# MAGIC A frontier model (Claude, through the Foundation Model API) reads the weights of Qwen3.5-0.8B, probes it with a
# MAGIC few forward passes, and writes a sparse weight patch. GEPA's Pareto engine decides which patch to keep.
# MAGIC Nothing is trained. Four rows come out:
# MAGIC
# MAGIC 1. base model, blog sampling (T=0.6)
# MAGIC 2. base model, Qwen model-card sampling (T=1.0) — the decoding baseline critics asked for
# MAGIC 3. the edit Astra published (131k weights, sha256-verified rebuild)
# MAGIC 4. the edit the optimizer finds here, at notebook budget
# MAGIC
# MAGIC Budget is far below the blog's (50 passes at 32k tokens, 20 h). Read the numbers as "does the loop work",
# MAGIC not as a MATH-500 claim.

# COMMAND ----------

# MAGIC %pip install -q git+https://github.com/Praneeth16/dspy-weightspace.git

# COMMAND ----------

# Optional fast kernels for the Gated DeltaNet layers; the reference PyTorch path is used if this fails.
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flash-linear-attention"], check=False)
dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import time

import mlflow
import torch

from dspy_weightspace import HFLocalLM, MathSolver, WeightSpaceOptimizer, load_gsm8k, math_metric
from dspy_weightspace.baseline import results_table, score_rows
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.reflection import databricks_lm

dbutils.widgets.text("reflection_endpoint", "databricks-claude-sonnet-5-5")
dbutils.widgets.text("catalog", "serverless_lakebase_praneeth_catalog")
dbutils.widgets.text("schema", "weightspace")
dbutils.widgets.text("n_dev", "48")
dbutils.widgets.text("n_holdout", "48")
dbutils.widgets.text("max_new_tokens", "4096")
dbutils.widgets.text("passes_per_round", "8")
dbutils.widgets.text("max_metric_calls", "100")
dbutils.widgets.text("max_turns", "30")

P = {k: dbutils.widgets.get(k) for k in ["reflection_endpoint", "catalog", "schema"]}
N_DEV, N_HOLD = int(dbutils.widgets.get("n_dev")), int(dbutils.widgets.get("n_holdout"))
MAX_NEW, PASSES, METRIC_CALLS = (
    int(dbutils.widgets.get(k)) for k in ["max_new_tokens", "passes_per_round", "max_metric_calls"]
)
print(P, N_DEV, N_HOLD, MAX_NEW, PASSES, METRIC_CALLS, "cuda:", torch.cuda.is_available())

# COMMAND ----------

user = spark.sql("select current_user()").first()[0]
spark.sql(f"create schema if not exists {P['catalog']}.{P['schema']}")
spark.sql(f"create volume if not exists {P['catalog']}.{P['schema']}.artifacts")
VOL = f"/Volumes/{P['catalog']}/{P['schema']}/artifacts"
RUN_DIR = f"{VOL}/runs/{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=True)
mlflow.set_experiment(f"/Users/{user}/dspy-weightspace")

# COMMAND ----------

lm = HFLocalLM(revision=BASE_REVISION, batch_size=16)
prog = MathSolver(lm, max_new_tokens=MAX_NEW)
dev, hold = load_gsm8k(n_dev=N_DEV, n_holdout=N_HOLD)
print(lm.device, lm.dtype, type(lm.hf).__name__, len(lm.names), "tensors")

# COMMAND ----------

# MAGIC %md ## Baselines and the published edit

# COMMAND ----------

rows = {}
t0 = time.time()
rows["base, T=0.6 (blog sampling)"] = score_rows(prog, hold, math_metric)
print(f"{time.time() - t0:.0f}s per {N_HOLD}-problem pass")
rows["base, T=1.0 (Qwen card)"] = score_rows(prog, hold, math_metric, temperature=1.0)

NAMES = ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"]
published = build({n: lm.original(n) for n in NAMES}, cache_dir=f"{VOL}/exp8_edit")
lm.apply_patch(published)
rows["published Exp-8 edit (131k weights)"] = score_rows(prog, hold, math_metric)
lm.reset()
print(results_table(rows))

# COMMAND ----------

# MAGIC %md ## Let Claude find its own edit

# COMMAND ----------

reflection = databricks_lm(P["reflection_endpoint"])
opt = WeightSpaceOptimizer(
    math_metric,
    reflection,
    max_metric_calls=METRIC_CALLS,
    passes_per_round=PASSES,
    max_items_per_pass=24,
    reflection_minibatch_size=8,
    max_turns=MAX_TURNS,
    log_dir=f"{RUN_DIR}/optimizer",
)
with mlflow.start_run(run_name="weightspace-optimizer") as run:
    mlflow.log_params(
        {
            "model": lm.model_name,
            "reflection": P["reflection_endpoint"],
            "n_dev": N_DEV,
            "n_holdout": N_HOLD,
            "max_new_tokens": MAX_NEW,
            "passes_per_round": PASSES,
            "max_metric_calls": METRIC_CALLS,
        }
    )
    t0 = time.time()
    optimized = opt.compile(prog, trainset=dev, valset=dev)
    wall = time.time() - t0
    rows["optimizer-found edit"] = score_rows(optimized, hold, math_metric)
    lm.reset()
    mlflow.log_text(optimized.patch_src, "patch.py")
    mlflow.log_metrics(
        {
            "weights_changed": optimized.weights_changed,
            "wall_seconds": wall,
            "gepa_metric_calls": optimized.detailed_results.total_metric_calls or 0,
            **{f"holdout_{k}_{m}": v[m] for k, v in rows.items() for m in ("accuracy", "cut_rate")},
        }
    )
    mlflow.log_artifacts(f"{RUN_DIR}/optimizer", "optimizer")
    table = results_table(rows)
    mlflow.log_text(table, "results.md")
print(table)
print(
    "edited tensors:",
    optimized.edited_tensors,
    "weights changed:",
    optimized.weights_changed,
    f"wall {wall / 60:.0f} min",
)
print(optimized.patch_src)

# COMMAND ----------

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
dbutils.notebook.exit(json.dumps(out))
