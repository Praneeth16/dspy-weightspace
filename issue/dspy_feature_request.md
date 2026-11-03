# [Feature] Weight-space programming as an optimizer: let a reflection LM patch the student LM's weights

> Paste into https://github.com/stanfordnlp/dspy/issues/new?template=feature_request.yml
> Tick "Yes, I'd like to help implement this."

## What feature would you like to see?

An optimizer whose search space is the weights of a small local model used by a DSPy program, not its prompt and
not a full finetune. The MIT CSAIL post "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, 2 Nov 2026)
showed a frontier model reading Qwen3.5-0.8B's weights, finding the direction that drives reasoning loops, and
editing 0.016% of the weights to fix it, with no gradients.

That loop has the same shape as GEPA: evaluate, reflect on failures, propose, keep the Pareto front. The component
GEPA mutates is a string. A weight patch is also a string: a short Python script that edits named tensors. So the
GEPA engine can drive it unchanged.

Proposed shape, one rung between the two existing ones:

| Optimizer | What changes | Cost | Reversible |
|---|---|---|---|
| `dspy.GEPA` | instruction text | LLM calls | yes |
| `WeightSpaceOptimizer` (proposed) | sparse weight patch | LLM calls + a few forward passes | yes, patch is a diff |
| `dspy.BootstrapFinetune` | all weights | GPU training job | no |

```python
lm = HFLocalLM("Qwen/Qwen3.5-0.8B")  # in-process transformers, keeps a CPU copy of original tensors
student = MathSolver(lm)
opt = WeightSpaceOptimizer(metric, reflection_lm=dspy.LM("databricks/databricks-claude-..."), passes_per_round=8)
optimized = opt.compile(student, trainset=dev, valset=dev)
optimized.patch_src  # standalone script that reproduces the edit from original weights
optimized.weights_changed  # count
```

Pieces:
- `HFLocalLM`: in-process HF model with `apply_patch` / `reset`, plus batched generation that keeps `finish_reason`
  (termination is the signal for loops). Implemented on the legacy `BaseLM.forward` for 3.4; would move to an `Engine`.
- `PatchSandbox`: persistent kernel exposing the blog's tool surface (`load`, `load_original`, `save_edit`,
  `drop_edit`, `edited`, `tok`), enforcing the blog's scope rules (no embeddings, output head, final norm, or
  whole-tensor scaling; no gradients).
- `Probes`: the blog's budgeted forward-pass tool (`generate` with paired original/edited comparison and residual-stream
  activations; `teacher_forced`).
- `WeightSpaceAdapter(GEPAAdapter)`: candidate `{"weight_patch": <script>}`; `evaluate` runs the script in a fresh
  sandbox, applies, runs the program, resets; `make_reflective_dataset` gives failing outputs + feedback.
- `WeightPatchProposer(ProposalFn)`: the reflection LM's multi-turn session (code blocks, probe requests, final patch).
- `WeightSpaceOptimizer(Teleprompter)`: wires the above into `gepa.optimize(...)`, Pareto selection, `max_metric_calls`.

Working prototype as a standalone package, with a Colab notebook (T4) and a Databricks notebook:
https://github.com/Praneeth16/dspy-weightspace

Results at notebook budget (GSM8K subset, 2k tokens, 8 passes per round), holdout n=48:

| Variant | Accuracy | Looping | Cut at limit | Mean tokens | n |
|---|---|---|---|---|---|
| base, T=0.6 (blog sampling) | 16.7% | 62.5% | 97.9% | 4027 | 48 |
| base, T=1.0 (Qwen model card) | 8.3% | 0.0% | 100.0% | 4096 | 48 |
| published Experiment-8 edit (131k weights, sha256-verified) | 27.1% | 18.8% | 89.6% | 3845 | 48 |
| optimizer, cold start (Claude Sonnet 5.5, 4 rounds x 4 passes) | no edit submitted | | | | |
| optimizer, warm start from the published edit (6 passes/round) | running | | | | |

Run: Databricks AI Runtime, one A10 (24 GB), Qwen3.5-0.8B bf16, GSM8K holdout n=48, thinking on, 4,096 new tokens,
T=0.6 / top_p 0.95 / top_k 20, seed 0. "Looping" = distinct 8-gram ratio over the tail below 0.6. One 48-problem pass took
about 9 minutes with the reference Gated DeltaNet kernels.

What the cold-start optimizer did (transcripts in the MLflow run): in every round Claude built a repetition direction
(difference of mean residuals between repetitive and productive continuations, the blog's own method), projected it
out of mid-layer `down_proj` columns, measured the edit paired against the original on fresh problems, saw no gain
(looping 0.67 to 0.63, accuracy down), and declined to submit it. Four passes per round against the blog's fifty.

The published Experiment-8 edit is rebuilt from the authors' logs and sha256-verified against the evaluated checkpoint,
so the notebook also answers the decoding-baseline question (T=1.0 per the Qwen model card) in the same table.

## Why in DSPy rather than only a package

- `compile()` symmetry with GEPA and BootstrapFinetune; `BetterTogether` could chain prompt → patch → finetune.
- The reflective dataset, metric shape and Pareto engine are already DSPy's. The new code is the adapter, the sandbox
  and the local LM, about 600 lines.
- The authors say a benchmark and harness are in progress. If that lands in-tree, happy to fold this into it instead.

## Would you like to contribute?

- [x] Yes, I'd like to help implement this.

## Additional Context

- Post: https://rikiyat.github.io/blog/weight-space-programming/
- Logs with the exact edit: https://github.com/RikiyaT/lm_understand-logs (exp8_math)
- Rebuilt checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming

**AI disclosure (per CONTRIBUTING):** I used Claude Code to draft the prototype and this text; I reviewed and ran every
piece. Origin: my own experiment reproducing the blog post, not an issue sweep. Prompts used are in the repo under
`issue/prompts.md`.
