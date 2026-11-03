"""Small live run of the optimizer with a real reflection LM, to shake out the proposer loop before a long GPU job.

DATABRICKS_CONFIG_PROFILE=<profile> uv run python scripts/live_mini.py
"""

from __future__ import annotations

import os
import time

from dspy_weightspace import HFLocalLM, MathSolver, WeightSpaceOptimizer, load_gsm8k, math_metric
from dspy_weightspace.baseline import results_table, score_rows
from dspy_weightspace.published_edit import BASE_REVISION
from dspy_weightspace.reflection import databricks_lm

N, MAX_NEW = int(os.environ.get("N", 6)), int(os.environ.get("MAX_NEW", 1024))
lm = HFLocalLM(revision=BASE_REVISION, batch_size=N)
prog = MathSolver(lm, max_new_tokens=MAX_NEW)
dev, hold = load_gsm8k(n_dev=N, n_holdout=N)
reflection = databricks_lm(
    os.environ.get("ENDPOINT", "databricks-claude-sonnet-5-5"), profile=os.environ.get("DATABRICKS_CONFIG_PROFILE")
)

rows = {"base": score_rows(prog, hold, math_metric)}
opt = WeightSpaceOptimizer(
    math_metric,
    reflection,
    max_metric_calls=int(os.environ.get("CALLS", 30)),
    passes_per_round=2,
    max_items_per_pass=8,
    reflection_minibatch_size=3,
    max_turns=16,
    log_dir="runs/live_mini",
)
t0 = time.time()
out = opt.compile(prog, trainset=dev, valset=dev)
print(
    f"\ncompile done in {(time.time() - t0) / 60:.1f} min; candidates={len(out.detailed_results.candidates)} "
    f"metric_calls={out.detailed_results.total_metric_calls}"
)
rows["optimizer"] = score_rows(out, hold, math_metric)
lm.reset()
print(results_table(rows))
print("edited:", out.edited_tensors, "changed:", out.weights_changed)
print("PATCH:\n" + out.patch_src)
