# dspy-weightspace

Weight-space programming as a DSPy optimizer. A frontier model reads a small local model's weights, probes it with a
handful of forward passes, and writes a sparse patch. GEPA's Pareto engine decides which patch to keep. Nothing is
trained.

Built on the MIT CSAIL post ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab, 2 Nov 2026), where GPT-6 Astra edited 0.016% of Qwen3.5-0.8B's weights and stopped it
looping in its reasoning. Their loop has GEPA's shape: evaluate, reflect, propose, keep the Pareto front. GEPA mutates a
string. A weight patch is also a string: a short script that edits named tensors. So the GEPA engine drives it unchanged.

| Optimizer | What changes | Cost | Reversible |
|---|---|---|---|
| `dspy.GEPA` | instruction text | LLM calls | yes |
| **`WeightSpaceOptimizer`** (this repo) | sparse weight patch | LLM calls + a few forward passes | yes, patch is a diff |
| `dspy.BootstrapFinetune` | all weights | GPU training job | no |

## Results

Filled in from the Databricks run. Holdout GSM8K subset, thinking on, 4,096 new tokens max. Score = 0.4 correct + 0.3 terminated + 0.3 not looping (distinct 8-gram ratio over the tail >= 0.6).

__RESULTS_TABLE__

Budget here is a small fraction of the blog's (50 passes at 32k tokens, 158 turns, 20 hours). Read the table as
"does the loop work", not as a MATH-500 claim. The published Experiment-8 edit is rebuilt from the authors' logs and
sha256-verified against the checkpoint they evaluated; the two rebuilt tensors are bit-identical to the ones in
[ML-Intern-lab/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/ML-Intern-lab/Qwen3.5-0.8B-weight-space-programming)
(the checkpoint Hugging Face's ML Intern rebuilt and scored at 51.0% MATH-500 with the full 32k-token protocol).
For that full protocol on a GPU box, `scripts/eval_vllm.py` reproduces it with vLLM. The T=1.0 row is the Qwen model-card sampling that critics said
should have been the baseline. Both sit in the same table as the optimizer's own edit.

## How it works

```python
from dspy_weightspace import HFLocalLM, MathSolver, WeightSpaceOptimizer, load_gsm8k, math_metric
from dspy_weightspace.reflection import databricks_lm

lm = HFLocalLM("Qwen/Qwen3.5-0.8B")  # in-process, keeps a CPU copy of the original tensors
student = MathSolver(lm, max_new_tokens=2048)
dev, holdout = load_gsm8k(48, 48)

opt = WeightSpaceOptimizer(
    math_metric, reflection_lm=databricks_lm("databricks-claude-sonnet-5-5"), passes_per_round=8, max_metric_calls=200
)
optimized = opt.compile(student, trainset=dev, valset=dev)

optimized.patch_src  # standalone script that reproduces the edit from original weights
optimized.weights_changed  # how many entries differ
optimized.edited_tensors  # which tensors
```

Pieces, in `dspy_weightspace/`:

- `hf_lm.py` `HFLocalLM`: in-process transformers model with `apply_patch` / `reset`, batched generation that keeps
  `finish_reason`, and residual-stream readout. Legacy `BaseLM.forward` for DSPy 3.4; `Engine` port is a follow-up.
- `sandbox.py` `PatchSandbox`: the blog's tool surface (`load`, `load_original`, `save_edit`, `drop_edit`, `edited`,
  `tok`) with its scope rules enforced: embeddings, output head and final norm are frozen, whole-tensor scaling is
  refused, gradients are off. Plus `save_artifact` / `load_artifact` so a final patch can reuse vectors found in exploration.
- `probes.py` `Probes`: budgeted forward passes. `generate` (original / edited / both with paired diff, optional
  activations) and `teacher_forced`.
- `proposer.py` `WeightPatchProposer`: the reflection model's session. System prompt adapted from the blog's. It replies
  with one ```python, ```forward or ```patch block per turn; the patch is validated in a fresh kernel before it becomes
  a GEPA candidate.
- `adapter.py` `WeightSpaceAdapter`: GEPA adapter whose only component is `weight_patch`.
- `optimizer.py` `WeightSpaceOptimizer`: `Teleprompter.compile()` wired to `gepa.optimize`, Pareto selection.
- `published_edit.py`: rebuilds the Experiment-8 edit from the public `edit.npz` + `manifest.json`, sha256-checked.
- `metrics.py`, `data.py`, `program.py`, `baseline.py`: GSM8K subset, the program under optimization, the table.

## Run it

Colab (free T4): [`notebooks/colab.ipynb`](notebooks/colab.ipynb). Needs `ANTHROPIC_API_KEY` in Colab secrets, or point
`reflection` at any Databricks Foundation Model API endpoint.

Databricks (serverless GPU job): [`databricks/README.md`](databricks/README.md). Reflection model auth comes from the
notebook context. Results land in MLflow and a UC volume.

Local:

```bash
uv sync --all-groups
uv run pytest
uv run python scripts/smoke_local.py --n 8 --max-new-tokens 2048   # loads the model, verifies the published edit, 3 rows
```

Apple silicon runs the 0.8B model on MPS. The Gated DeltaNet layers fall back to reference PyTorch kernels without
`flash-linear-attention`; correct, slower. On CUDA install `flash-linear-attention` for speed.

## Honest limits

- Removes obstacles, does not add skill. The blog's Gemma 3 270M follow-up, which rarely loops, got nothing. Expect the same.
- Open weights only. Nothing here applies to a model behind an API.
- Needs an eval gate. The blog's IFEval went 49 → 40 → 62 across 10/25/50 passes. Pareto selection over a valset is
  the gate here; keep the holdout separate.
- "Native weight reading" at 0.8B means tool-assisted inspection in a Python kernel, as in the blog's Experiment 8.
- HF `generate` has no presence penalty; the blog used 1.5. `repetition_penalty` is the nearest knob and is off by default.

## Status

Prototype, 3 Nov 2026. Feature request text for `stanfordnlp/dspy` is in [`issue/`](issue/). The authors say a benchmark
and harness are in progress; if that lands in-tree this folds into it.

MIT.
