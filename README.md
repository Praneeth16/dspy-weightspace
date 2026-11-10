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

`WeightSpaceOptimizer` is a DSPy `Teleprompter`. Its `compile()` runs the GEPA engine (`gepa.optimize`) through
`WeightSpaceAdapter`, whose only component is a weight-patch script. It does not use `dspy.GEPA`, which evolves
instruction text.

## Results

MATH-500, all 500 problems, 2 samples each, thinking on, 32,768 new tokens, the original evaluation's sampling
(T=0.6, top-p 0.95, top-k 20, presence penalty 1.5), strict grading (no closing think tag means no answer). Intervals
are paired 95% bootstrap intervals over problems, against the published edit.

| Variant | Accuracy | vs published | Tokens per correct answer |
|---|---|---|---|
| base | 7.1% | | 434k |
| base, thinking off | 45.6% | -7.7 [-10.9, -4.6] | 8.3k |
| published Experiment-8 edit (Astra) | 53.3% | | 40.5k |
| difference-in-means direction, same placement, no LLM | 29.8% | -23.5 | 93.6k |
| masked fine-tuning of the same 131k weights on GSM8K solutions | 47.6% | -5.7 [-9.0, -2.4] | 24.6k |
| **this optimizer, from scratch, best of 3 seeds** | **60.6%** | **+7.3 [4.6, 10.0]** | 31.3k |
| **this optimizer, from the published edit, best of 3 seeds** | **59.9%** | **+6.6 [3.9, 9.3]** | 29.6k |

Search setup for the last two rows: Claude Opus 5.5 as the reflection model, 160 metric calls, MATH train problems at
levels 3 to 5 (disjoint from MATH-500), 16,384-token selection with the test's sampling, generation in vLLM. All six
runs:

| Run | Selection score | Held-out (48) | MATH-500 |
|---|---|---|---|
| from scratch, seed 0 / 1 / 2 | 0.190 / 0.296 / 0.350 | 16.7% / 29.2% / 22.9% | 32.0% / 60.6% / 40.6% |
| from the published edit, seed 0 / 1 / 2 | 0.326 / 0.296 / 0.282 | 29.2% / 27.1% / 20.8% | 54.3% / 59.9% / 32.5% |

What to take from it:

1. **Run several seeds.** The same configuration scores anywhere from 32% to 61%. One run tells you little.
2. **Choose between runs with a held-out set that the search never sees.** Across the six runs, held-out accuracy has
   a rank correlation of 0.90 with MATH-500. The selection score that GEPA optimizes has 0.46. Within one run, every
   child candidate did at least as well on the held-out set as its parent, so the selection score is fine for steering
   one search. It is not fine for comparing searches. Choosing by held-out score picks 60.6% from scratch and 54.3% from the
   published edit, where seed 0 beat seed 1 on the held-out set by one problem.
3. **Selection data must look like the test.** Our first runs selected on short GSM8K problems and found edits that
   scored about 15% on MATH-500.
4. **The edits that work make the model conclude.** Every strong edit makes `</think>` and words like "So" and
   "Therefore" 4 to 7.5 times more likely at a sentence end. Weak edits mostly make "Wait" a little less likely.

The best edit from scratch removes one repetition direction from the output of every MLP and attention block in
layers 10 to 22. The best warm edit keeps the published edit and scales four "Wait"-suppressing columns by 1.5. Our
earlier GSM8K numbers (in `evidence/`) used a lenient grader that counted answers inside unfinished thinking. On
MATH-500 that grader raised the base model from 8.0% to 21.6%, so do not compare them with the table above.

## How it works

```python
from dspy_weightspace import HFLocalLM, MathSolver, WeightSpaceOptimizer, load_math, math_metric
from dspy_weightspace.reflection import databricks_lm

lm = HFLocalLM("Qwen/Qwen3.5-0.8B")  # in-process, keeps a CPU copy of the original tensors
student = MathSolver(lm, max_new_tokens=16384)

# Three disjoint sets. The reflection model probes and reads feedback on the first, GEPA selects on the second,
# and you choose between runs on the third. Keep the test set out of all three.
dev, holdout = load_math(n_dev=48, n_holdout=48)  # MATH train, levels 3 to 5
feedback, selection = dev[:16], dev[16:]

runs = []
for seed in (0, 1, 2):
    opt = WeightSpaceOptimizer(
        math_metric,
        reflection_lm=databricks_lm("databricks-claude-opus-5-5"),
        passes_per_round=8,
        max_metric_calls=160,
        seed=seed,
        log_dir=f"runs/seed{seed}",
    )
    runs.append(opt.compile(student, trainset=feedback, valset=selection))
    lm.reset()
# score each run's patch on `holdout` and keep the best

optimized = runs[0]
optimized.patch_src  # standalone script that reproduces the edit from original weights
optimized.weights_changed  # how many entries differ
optimized.edited_tensors  # which tensors
```

For long traces, pass `gen_backend=VLLMBackend(model_dir)` (`dspy_weightspace/vllm_backend.py`) to `HFLocalLM`. Patches
are copied into the vLLM engine in place, and generation is about ten times faster than Hugging Face `generate` on
this model. `databricks/run_air.py` is the full script behind the table, including scoring each candidate on the
held-out set after the search ends.

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
- `vllm_backend.py` `VLLMBackend`: fast generation for selection, with in-place weight copies and prefix caching off.
- `steering.py`, `masked_ft.py`: the two baselines without a language model (difference-in-means direction, masked fine-tuning).
- `optimizer.py` `WeightSpaceOptimizer`: `Teleprompter.compile()` wired to `gepa.optimize`, Pareto selection.
- `published_edit.py`: rebuilds the Experiment-8 edit from the public `edit.npz` + `manifest.json`, sha256-checked.
- `metrics.py`, `data.py`, `program.py`, `baseline.py`: strict grading with math-verify, GSM8K and MATH subsets, the program under optimization, the table.

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
- Needs an eval gate. The blog's IFEval went 49 → 40 → 62 across 10/25/50 passes. Here, results swing from 32% to
  61% between seeds of the same search. Keep a held-out set apart from both `trainset` and `valset`.
- One model, one failure mode (thinking loops), one test set, one reflection model, three seeds per setup.
- "Native weight reading" at 0.8B means tool-assisted inspection in a Python kernel, as in the blog's Experiment 8.
- HF `generate` has no presence penalty; the blog used 1.5. Use the vLLM backend, which applies it, to match the test.

## Status

Prototype, updated 10 Nov 2026. Feature request text for `stanfordnlp/dspy` is in [`issue/`](issue/). Related work:
Drew Breunig's [`dspy-wsp`](https://pypi.org/project/dspy-wsp/) packages weight-space programming as a DSPy optimizer
too, and Matin Khajavi's [replication](https://github.com/MatinKhajavi/Weight-Space-Programming) reruns all eight
experiments of the original post with Claude Opus 5.5.

MIT.
