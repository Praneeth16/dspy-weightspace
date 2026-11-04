"""Track B on AI Runtime: decision-typed weight edits, tuned by dspy.GEPA and calibrated by ReAnchor.

Env: WS_VOLUME, DECISION_ENDPOINT (LM that makes the decisions), REFLECTION_ENDPOINT (GEPA's reflection LM),
N_DEV, N_HOLDOUT, MAX_NEW_TOKENS, GEPA_CALLS, SUBSET.
"""

from __future__ import annotations

import json
import os
import time

import dspy
import mlflow
import torch

from dspy_weightspace import HFLocalLM, MathSolver, load_gsm8k, math_metric
from dspy_weightspace.baseline import results_table, score_rows
from dspy_weightspace.decision_editor import DecisionEditor, EditBuilder, diagnostics_text
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.reflection import databricks_lm
from dspy_weightspace.sandbox import PatchSandbox

E = os.environ
VOL = E["WS_VOLUME"]
DECISION_EP = E.get("DECISION_ENDPOINT", "databricks-claude-sonnet-5-5")
REFLECTION_EP = E.get("REFLECTION_ENDPOINT", "databricks-claude-opus-5-5")
N_DEV, N_HOLD = int(E.get("N_DEV", 32)), int(E.get("N_HOLDOUT", 48))
MAX_NEW, GEPA_CALLS, SUBSET = int(E.get("MAX_NEW_TOKENS", 8192)), int(E.get("GEPA_CALLS", 40)), int(E.get("SUBSET", 8))
RUN_DIR = f"{VOL}/runs/{time.strftime('%Y%m%d-%H%M%S')}-trackb"
os.makedirs(RUN_DIR, exist_ok=True)
print(
    {
        "decision": DECISION_EP,
        "reflection": REFLECTION_EP,
        "n_dev": N_DEV,
        "n_holdout": N_HOLD,
        "max_new": MAX_NEW,
        "gepa_calls": GEPA_CALLS,
        "subset": SUBSET,
        "run_dir": RUN_DIR,
        "cuda": torch.cuda.is_available(),
    },
    flush=True,
)
mlflow.set_tracking_uri("databricks")

lm = HFLocalLM(revision=BASE_REVISION, batch_size=16)
prog = MathSolver(lm, max_new_tokens=MAX_NEW)
dev, hold = load_gsm8k(n_dev=N_DEV, n_holdout=N_HOLD)
NAMES = ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"]
published = build({n: lm.original(n) for n in NAMES}, cache_dir=f"{VOL}/exp8_edit")
builder = EditBuilder(lm, published)
dspy.configure(lm=databricks_lm(DECISION_EP))

# ----- situations: each dev subset is one training example for the decision program ---------------------------
subsets = [dev[i : i + SUBSET] for i in range(0, len(dev), SUBSET)]
situations = []
for k, sub in enumerate(subsets):
    rows = {"original weights": score_rows(prog, sub, math_metric)}
    lm.apply_patch(published)
    rows["published edit at 1.0x (scale_published, s=1)"] = score_rows(prog, sub, math_metric)
    lm.reset()
    situations.append(dspy.Example(diagnostics=diagnostics_text(rows), subset=k).with_inputs("diagnostics"))
    print(f"situation {k}: {rows}", flush=True)


def edit_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Apply the decided edit to the model and score the decision's own subset. Accuracy first."""
    sub = subsets[gold.subset]
    lm.reset()
    if pred.edits:
        lm.apply_patch(pred.edits)
    r = score_rows(prog, sub, math_metric)
    lm.reset()
    d = pred.decision
    fb = (
        f"Decision {d.method} / {d.layers} / {d.strength:.2f}x / ship={d.ship}: accuracy {r['accuracy']:.1%}, "
        f"looping {r['loop_rate']:.1%}, cut {r['cut_rate']:.1%} on {len(sub)} problems."
    )
    return dspy.Prediction(score=r["score"], feedback=fb)


editor = DecisionEditor(builder)
with mlflow.start_run() as run:
    mlflow.log_params(
        {
            "track": "B-decision-types",
            "decision_lm": DECISION_EP,
            "reflection_lm": REFLECTION_EP,
            "n_dev": N_DEV,
            "n_holdout": N_HOLD,
            "max_new_tokens": MAX_NEW,
            "gepa_calls": GEPA_CALLS,
        }
    )
    t0 = time.time()
    gepa = dspy.GEPA(
        metric=edit_metric,
        reflection_lm=databricks_lm(REFLECTION_EP),
        max_metric_calls=GEPA_CALLS,
        reflection_minibatch_size=2,
        num_threads=1,
        track_stats=True,
        log_dir=f"{RUN_DIR}/gepa",
    )
    tuned = gepa.compile(editor, trainset=situations, valset=situations)
    t_gepa = time.time() - t0
    try:
        from dspy.experimental import ReAnchor

        calibrated = ReAnchor(edit_metric, require_cache=False, log_dir=f"{RUN_DIR}/reanchor").compile(
            tuned, trainset=situations
        )
        reanchor_report = calibrated.decide.fields if hasattr(calibrated, "decide") else {}
    except Exception as e:  # noqa: BLE001
        print("ReAnchor skipped:", e, flush=True)
        calibrated, reanchor_report = tuned, {"error": str(e)}
    wall = time.time() - t0

    # Final decision on the whole dev set's diagnostics, scored on the holdout.
    rows = {"base, T=0.6 (blog sampling)": score_rows(prog, hold, math_metric)}
    lm.apply_patch(published)
    rows["published Exp-8 edit (131k weights)"] = score_rows(prog, hold, math_metric)
    lm.reset()
    full_rows = {"original weights": score_rows(prog, dev, math_metric)}
    lm.apply_patch(published)
    full_rows["published edit at 1.0x (scale_published, s=1)"] = score_rows(prog, dev, math_metric)
    lm.reset()
    final = calibrated(diagnostics=diagnostics_text(full_rows))
    lm.reset()
    if final.edits:
        lm.apply_patch(final.edits)
    rows["Track B decision editor"] = score_rows(prog, hold, math_metric)
    changed, edited = lm.changed_entries(), lm.edited()
    lm.reset()

    sb = PatchSandbox(lm, RUN_DIR)
    sb.edits = dict(final.edits)
    patch_src = sb.export_script() if final.edits else ""
    table = results_table(rows)
    mlflow.log_text(final.decision.to_json(), "decision.json")
    mlflow.log_text(patch_src or "# no edit", "patch.py")
    mlflow.log_text(table, "results.md")
    mlflow.log_text(json.dumps(reanchor_report, default=str), "reanchor_fields.json")
    mlflow.log_text(tuned.decide.signature.instructions, "tuned_instruction.txt")
    mlflow.log_metrics(
        {
            "weights_changed": changed,
            "wall_seconds": wall,
            "gepa_seconds": t_gepa,
            **{
                f"holdout_{i}_{m}": v[m]
                for i, v in enumerate(rows.values())
                for m in ("accuracy", "cut_rate", "loop_rate")
            },
        }
    )
    if os.path.isdir(f"{RUN_DIR}/gepa"):
        mlflow.log_artifacts(f"{RUN_DIR}/gepa", "gepa")

print(table, flush=True)
print("decision:", final.decision.to_json(), flush=True)
print("tuned instruction:", tuned.decide.signature.instructions[:600], flush=True)
print("edited:", edited, "changed:", changed, f"wall {wall / 60:.0f} min", flush=True)
out = {
    "rows": rows,
    "decision": json.loads(final.decision.to_json()),
    "weights_changed": changed,
    "edited_tensors": edited,
    "wall_seconds": wall,
    "reanchor_fields": reanchor_report,
    "run_dir": RUN_DIR,
    "mlflow_run_id": run.info.run_id,
}
for name, content in [
    ("summary.json", json.dumps(out, indent=2, default=str)),
    ("results.md", table + "\n"),
    ("patch.py", patch_src or "# no edit\n"),
    ("decision.json", final.decision.to_json()),
]:
    with open(f"{RUN_DIR}/{name}", "w") as f:
        f.write(content)
print("SUMMARY " + json.dumps(out, default=str), flush=True)
