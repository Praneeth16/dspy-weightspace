"""No-LLM baselines on AI Runtime, under Track A's protocol: built on the feedback problems, selected on the selection
problems, scored once on the holdout. Each pick is written as a replayable run dir (patch.py + optimizer/artifacts)
for run_air_vllm.py.

BASELINE=diffmean  difference-in-means loop direction, edit W' = (I - a*u*u^T) W over a grid of layer sets and a,
                   plus a random-direction control at the chosen setting.
BASELINE=maskedft  masked SFT of the published edit's support (and of its two whole tensors) on the base model's own
                   correct traces, and on GSM8K rationales.

Env: WS_VOLUME, BASELINE, N_DEV, N_FEEDBACK, N_HOLDOUT, MAX_NEW_TOKENS, ALPHAS, LAYER_SETS, DIR_SAMPLES, SELF_SAMPLES,
SELF_MAX_NEW, SFT_LR, SFT_CHECKPOINTS, MLFLOW_TRACKING_URI.
"""

from __future__ import annotations

import json
import os
import time

import mlflow
import numpy as np

from dspy_weightspace import HFLocalLM, MathSolver, load_gsm8k, math_metric
from dspy_weightspace.baseline import results_table, score_rows
from dspy_weightspace.masked_ft import gold_traces, masked_sft, self_traces, support_mask
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.sandbox import PatchSandbox, run_fresh
from dspy_weightspace.steering import ablation_edits, ablation_script, loop_directions, random_like

E = os.environ
VOL, BASELINE = E["WS_VOLUME"], E["BASELINE"]
N_DEV, N_FEEDBACK, N_HOLD = int(E.get("N_DEV", 48)), int(E.get("N_FEEDBACK", 16)), int(E.get("N_HOLDOUT", 48))
MAX_NEW = int(E.get("MAX_NEW_TOKENS", 4096))
ALPHAS = [float(a) for a in E.get("ALPHAS", "1,2,3,6").split(",")]
LAYER_SETS = [
    [int(x) for x in s.split(",")] for s in E.get("LAYER_SETS", "15,19;12,13,14,15,16,17,18,19,20,21,22").split(";")
]
DIR_SAMPLES, SELF_SAMPLES = int(E.get("DIR_SAMPLES", 4)), int(E.get("SELF_SAMPLES", 8))
SELF_MAX_NEW, SFT_LR = int(E.get("SELF_MAX_NEW", 8192)), float(E.get("SFT_LR", 1e-4))
SFT_CHECKPOINTS = tuple(int(x) for x in E.get("SFT_CHECKPOINTS", "25,50,100,200").split(","))
STAMP = f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{BASELINE}"
RUN_DIR = f"{VOL}/{STAMP}"
os.makedirs(RUN_DIR, exist_ok=True)

mlflow.set_tracking_uri(E.get("MLFLOW_TRACKING_URI", "databricks"))
lm = HFLocalLM(revision=BASE_REVISION, batch_size=16)
prog = MathSolver(lm, max_new_tokens=MAX_NEW)
dev, hold = load_gsm8k(n_dev=N_DEV, n_holdout=N_HOLD)
feedback, select = dev[:N_FEEDBACK], dev[N_FEEDBACK:]
if not feedback or not select:
    raise ValueError(f"N_DEV={N_DEV} must exceed N_FEEDBACK={N_FEEDBACK}: both sets need problems")
PUB_NAMES = [
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
]
published = build({n: lm.original(n) for n in PUB_NAMES}, cache_dir=f"{VOL}/exp8_edit")
print({"baseline": BASELINE, "run_dir": RUN_DIR, "feedback": len(feedback), "select": len(select), "hold": len(hold)})


def score(edits, examples):
    lm.reset()
    if edits:
        lm.apply_patch(edits)
    r = score_rows(prog, examples, math_metric)
    changed = lm.changed_entries()
    lm.reset()
    return {**r, "weights_changed": changed}


def write_run(label, src, artifacts, meta, expect):
    """A run dir run_air_vllm.py can replay; the patch is re-run here and must rebuild the same edit."""
    d = f"{RUN_DIR}/{label}"
    sb = PatchSandbox(lm, f"{d}/optimizer")
    for k, v in artifacts.items():
        sb.save_artifact(k, v)
    if src is None:  # whole tensors: freeze them as artifacts
        sb.edits = dict(expect)
        src = sb.export_script()
    with open(f"{d}/patch.py", "w") as f:
        f.write(src)
    fresh, res = run_fresh(lm, src, f"{d}/optimizer")
    assert res.ok, res.error
    assert all(np.allclose(fresh.edits[n].numpy(), expect[n].numpy(), atol=1e-6) for n in expect), label
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return f"{STAMP}/{label}"


summary = {"baseline": BASELINE, "selection": [], "holdout": {}, "runs": {}}
with mlflow.start_run():
    mlflow.log_params({"baseline": BASELINE, "n_dev": N_DEV, "n_feedback": N_FEEDBACK, "max_new_tokens": MAX_NEW})
    summary["holdout"]["base"] = score({}, hold)
    summary["holdout"]["published"] = score(published, hold)

    if BASELINE == "diffmean":
        hidden = sorted({layer + 1 for s in LAYER_SETS for layer in s})
        dirs, counts = loop_directions(lm, prog, feedback, hidden, MAX_NEW, DIR_SAMPLES)
        summary["direction_counts"] = counts
        print("direction classes", counts, flush=True)
        for k, u in dirs.items():
            np.save(f"{RUN_DIR}/dirs_L{k}.npy", u.numpy())
        for layers in LAYER_SETS:
            for alpha in ALPHAS:
                r = score(ablation_edits(lm, dirs, layers, alpha), select)
                summary["selection"].append({"layers": layers, "alpha": alpha, **r})
                print("select", layers, alpha, r, flush=True)
        best = max(summary["selection"], key=lambda x: (x["score"], x["accuracy"]))
        rand = random_like(dirs, seed=0)
        for label, d in (("diffmean", dirs), ("randomdir", rand)):
            edits = ablation_edits(lm, d, best["layers"], best["alpha"])
            summary["holdout"][label] = score(edits, hold)
            src, arts = ablation_script(d, best["layers"], best["alpha"], label)
            meta = {"layers": best["layers"], "alpha": best["alpha"], "counts": counts}
            summary["runs"][label] = write_run(label, src, arts, meta, edits)

    elif BASELINE == "maskedft":
        self_pairs = self_traces(prog, feedback, SELF_SAMPLES, SELF_MAX_NEW)
        summary["self_traces"] = len(self_pairs)
        print(f"{len(self_pairs)} self traces kept of {SELF_SAMPLES * len(feedback)}", flush=True)
        gold_pairs = gold_traces(feedback)
        configs = [("ft-self", "published", self_pairs), ("ft-gold", "published", gold_pairs)]
        configs.append(("ft-gold-layers", "layers", gold_pairs))
        for label, support, pairs in configs:
            if len(pairs) < 4:
                summary["runs"][label] = f"skipped: {len(pairs)} training traces"
                continue
            masks = support_mask(lm, support, published)
            snaps, losses = masked_sft(lm, pairs, masks, lr=SFT_LR, checkpoints=SFT_CHECKPOINTS)
            rows = []
            for step, edits in snaps.items():
                r = score(edits, select)
                rows.append({"label": label, "support": support, "step": step, "loss": losses[step - 1], **r})
                print("select", rows[-1], flush=True)
            summary["selection"] += rows
            pick = max(rows, key=lambda x: (x["score"], x["accuracy"]))
            edits = snaps[pick["step"]]
            summary["holdout"][label] = score(edits, hold)
            meta = {"support": support, "step": pick["step"], "lr": SFT_LR, "traces": len(pairs), "losses": losses}
            summary["runs"][label] = write_run(label, None, {}, meta, edits)
    else:
        raise ValueError(f"BASELINE={BASELINE!r}")

    table = results_table(summary["holdout"])
    print(table, flush=True)
    mlflow.log_text(table, "results.md")
    mlflow.log_text(json.dumps(summary, indent=2, default=str), "summary.json")
with open(f"{RUN_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
with open(f"{RUN_DIR}/results.md", "w") as f:
    f.write(table + "\n")
print("SUMMARY " + json.dumps(summary, default=str), flush=True)
