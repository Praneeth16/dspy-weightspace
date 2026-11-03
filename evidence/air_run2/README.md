# AI Runtime run 2 (cold start), 3 Nov 2026

Databricks AI Runtime, one A10, Qwen3.5-0.8B, reflection model `databricks-claude-sonnet-5-5`, 32 dev / 48 holdout GSM8K,
4,096 new tokens, 4 passes per round, 80 GEPA metric calls. Four proposer rounds; rounds 2 and 4 were lost to a CUDA
out-of-memory in the activation probe (fixed in the next commit). Rounds 1 and 3 ran to completion and each ended with
an empty patch: the model measured its edit paired against the original and declined to submit it.

- `round_01_transcript.md`, `round_03_transcript.md`: every turn, code block, probe request and result.
- `results.md`: the holdout table. `summary.json`: machine-readable summary, MLflow run id.
