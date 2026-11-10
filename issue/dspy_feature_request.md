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
dev, holdout = load_math(n_dev=48, n_holdout=48)
feedback, selection = dev[:16], dev[16:]  # the reflection LM reads feedback; GEPA selects; holdout picks between runs
opt = WeightSpaceOptimizer(metric, reflection_lm=dspy.LM("databricks/databricks-claude-opus-5-5"), seed=0)
optimized = opt.compile(student, trainset=feedback, valset=selection)
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
- `WeightSpaceOptimizer(Teleprompter)`: wires the above into `gepa.optimize(...)` (the standalone GEPA engine, not
  `dspy.GEPA`), Pareto selection, `max_metric_calls`, `seed`.
- `VLLMBackend`: optional fast generation for selection, with patches copied into the engine in place.

Working prototype as a standalone package, with a Colab notebook (T4) and a Databricks notebook:
https://github.com/Praneeth16/dspy-weightspace

Results on the full MATH-500 protocol (500 problems, 2 samples each, 32,768 tokens, the original evaluation's sampling,
strict grading). Intervals are paired 95% bootstrap intervals against the published edit.

| Variant | MATH-500 | vs published |
|---|---|---|
| base | 7.1% | |
| base, thinking off | 45.6% | -7.7 [-10.9, -4.6] |
| published Experiment-8 edit | 53.3% | |
| masked fine-tuning of the same 131k weights on GSM8K solutions | 47.6% | -5.7 [-9.0, -2.4] |
| `WeightSpaceOptimizer`, from scratch (3 seeds) | 32.0% / 60.6% / 40.6% | best +7.3 [4.6, 10.0] |
| `WeightSpaceOptimizer`, from the published edit (3 seeds) | 54.3% / 59.9% / 32.5% | best +6.6 [3.9, 9.3] |

Reflection model Claude Opus 5.5, 160 metric calls per run, selection on MATH train problems (levels 3 to 5) at
16,384 tokens with the test's sampling. Full tables and scripts are in the repo.

Three findings bear on the optimizer API:

1. **`valset` is not a held-out set.** GEPA optimizes against it, so it ranks candidates within one run well but ranks
   separate runs poorly. Across our six runs its score has a rank correlation of 0.46 with MATH-500, while a 48-problem
   set that the search never saw has 0.90. The docstring and the example therefore use three disjoint sets.
2. **Seeds matter a lot.** The same configuration scores 32% to 61% on MATH-500. A `seed` argument and a documented
   "run several, choose on held-out" pattern seem necessary for any weight-space optimizer.
3. **Selection data must match the test.** Our first runs selected on short GSM8K problems with a metric that gave
   0.6 of its weight to finishing and not looping. They found edits that scored about 15% on MATH-500.

Separately, `dspy.GEPA`'s default instruction proposer uses `JSONAdapter`, which sends `response_format`. Claude 5.x
endpoints reject that with HTTP 400, so every reflection call failed in our run that used `dspy.GEPA`. Passing
`InstructionProposer(adapter=dspy.ChatAdapter())` fixed it. Happy to file that as its own issue.

## Why in DSPy rather than only a package

- `compile()` symmetry with GEPA and BootstrapFinetune; `BetterTogether` could chain prompt → patch → finetune.
- The reflective dataset, metric shape and Pareto engine are already DSPy's. The new code is the adapter, the sandbox,
  the probes, the proposer session and the local LM, about 1,200 lines including the optional vLLM backend.
- The authors say a benchmark and harness are in progress. If that lands in-tree, happy to fold this into it instead.
- Drew Breunig's [`dspy-wsp`](https://pypi.org/project/dspy-wsp/) (8 Nov 2026) is another DSPy implementation. Happy to
  merge efforts with it; the results and the three-set split above may be the most useful part to carry over.

## Would you like to contribute?

- [x] Yes, I'd like to help implement this.

## Additional Context

- Post: https://rikiyat.github.io/blog/weight-space-programming/
- Logs with the exact edit: https://github.com/RikiyaT/lm_understand-logs (exp8_math)
- Rebuilt checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming

**AI disclosure (per CONTRIBUTING):** I used Claude Code to draft the prototype and this text; I reviewed and ran every
piece. Origin: my own experiment reproducing the blog post, not an issue sweep. Prompts used are in the repo under
`issue/prompts.md`.
