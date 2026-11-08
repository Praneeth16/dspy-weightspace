"""Build notebooks/colab.ipynb from cell sources. Run: uv run python notebooks/build_colab.py"""

from __future__ import annotations

import json
import pathlib

MD = "markdown"
PY = "code"

CELLS = [
    (
        MD,
        """# Weight-space programming as a DSPy optimizer

A frontier model reads the weights of Qwen3.5-0.8B, probes it with a few forward passes, and writes a sparse weight
patch. GEPA's Pareto engine decides which patch to keep. Nothing is trained.

Four rows come out: base at blog sampling (T=0.6), base at Qwen model-card sampling (T=1.0), the edit Astra published
(sha256-verified rebuild), and the edit the optimizer finds here. Budget is far below the blog's (50 passes at 32k tokens,
20 hours); read the numbers as "does the loop work", not as a MATH-500 claim.

Runtime: T4 GPU (free tier). Reflection model: Claude via the Anthropic API (`ANTHROPIC_API_KEY` in Colab secrets),
or any Databricks Foundation Model API endpoint (`DATABRICKS_HOST` + `DATABRICKS_TOKEN`).""",
    ),
    (PY, """!pip install -q git+https://github.com/Praneeth16/dspy-weightspace.git"""),
    (
        PY,
        """import os, time, torch
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

from dspy_weightspace import HFLocalLM, MathSolver, WeightSpaceOptimizer, load_gsm8k, math_metric
from dspy_weightspace.baseline import results_table, score_rows
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.reflection import anthropic_lm

N_DEV, N_HOLD, MAX_NEW, PASSES, METRIC_CALLS = 24, 24, 1536, 6, 120
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")""",
    ),
    (
        PY,
        """lm = HFLocalLM(revision=BASE_REVISION, batch_size=8)
prog = MathSolver(lm, max_new_tokens=MAX_NEW)
dev, hold = load_gsm8k(n_dev=N_DEV, n_holdout=N_HOLD)
print(lm.device, lm.dtype, type(lm.hf).__name__, len(lm.names), "tensors")""",
    ),
    (MD, "## Baselines and the published edit"),
    (
        PY,
        """rows = {}
t0 = time.time()
rows["base, T=0.6 (blog sampling)"] = score_rows(prog, hold, math_metric)
print(f"{time.time() - t0:.0f}s per {N_HOLD}-problem pass")
rows["base, T=1.0 (Qwen card)"] = score_rows(prog, hold, math_metric, temperature=1.0)

NAMES = ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"]
published = build({n: lm.original(n) for n in NAMES})
lm.apply_patch(published)
rows["published Exp-8 edit (131k weights)"] = score_rows(prog, hold, math_metric)
lm.reset()
print(results_table(rows))""",
    ),
    (MD, "## Let the reflection model find its own edit"),
    (
        PY,
        """reflection = anthropic_lm("anthropic/claude-fable-5-1")
opt = WeightSpaceOptimizer(math_metric, reflection, max_metric_calls=METRIC_CALLS, passes_per_round=PASSES,
                           max_items_per_pass=16, reflection_minibatch_size=6, log_dir="runs/colab")
t0 = time.time()
optimized = opt.compile(prog, trainset=dev[: len(dev) // 3], valset=dev[len(dev) // 3 :])
wall = time.time() - t0
rows["optimizer-found edit"] = score_rows(optimized, hold, math_metric)
lm.reset()
print(results_table(rows))
print("edited tensors:", optimized.edited_tensors, "weights changed:", optimized.weights_changed, f"wall {wall/60:.0f} min")
print(optimized.patch_src)""",
    ),
    (
        MD,
        """## What to read in the transcript

`runs/colab/round_01/transcript.md` is the full session: what the model asked the probes, what it computed from the
weights, and the patch it wrote. The patch is a standalone script. Run it in a fresh `PatchSandbox` to reproduce the edit.""",
    ),
]


def main() -> None:
    nb = {
        "cells": [
            {
                "cell_type": t,
                "metadata": {},
                "source": src.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if t == PY else {}),
            }
            for t, src in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = pathlib.Path(__file__).with_name("colab.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
