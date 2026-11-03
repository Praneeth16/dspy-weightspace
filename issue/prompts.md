# Prompts used (AI disclosure for the DSPy feature request)

Tool: Claude Code (Claude Fable 5.1), 3 Nov 2026. Everything below was reviewed and run by the author.

## Shaping the idea
- "What can be an unique perspective on this we can take and write an article on it? And do you think we have a
  possibility of this being implemented in enterprise settings?" (on the weight-space programming post)
- "how about integrating this technique in DSPy??"
- "can we raise a PR for this to DSPy???"
- "can you check DSPy contribution guidelines?"
- "Go ahead; and lets test it also end to end on lakebase praneeth workspace."

## Research the model did before coding
- Read DSPy 3.4.1 sources: `teleprompt/bootstrap_finetune.py`, `teleprompt/bootstrap_trace.py`,
  `teleprompt/gepa/*`, `clients/lm_local.py`, `clients/provider.py`, `clients/base_lm.py`, `evaluate/evaluate.py`,
  `tests/teleprompt/*`; gepa 0.1.4 `api.py`, `core/adapter.py`, `core/result.py`.
- Read `RikiyaT/lm_understand-logs/exp8_math`: `README.md`, `runs/p50c/turns/t001/prompt.txt`,
  `runs/p50c/edit/manifest.json`, `code/apply_edit.py`, `code/st_numpy.py`, `code/profiles.py`.
- Read `CONTRIBUTING.md`, issue and PR templates, and the PR-title CI check in `stanfordnlp/dspy`.

## Design brief given to the model (summary)
Build a `Teleprompter` whose GEPA candidate is a weight-patch script; reuse the blog's tool names and scope rules;
in-process HF model with reset; budgeted probes; GEPA Pareto selection; honest notebook budget; always include the
decoding baseline and the published edit in the same table.

The system prompt the reflection LM receives is `dspy_weightspace/proposer.py::SYSTEM`, adapted from the blog's
`prompt.txt`.
