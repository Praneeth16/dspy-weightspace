"""Local smoke test: load the model, check tensor names, rebuild the published edit, generate a few answers.

uv run python scripts/smoke_local.py [--n 4] [--max-new-tokens 512]
"""

from __future__ import annotations

import argparse
import time

from dspy_weightspace import HFLocalLM, MathSolver, load_gsm8k, math_metric
from dspy_weightspace.baseline import results_table, score_rows
from dspy_weightspace.published_edit import BASE_REVISION, build

NAMES = [
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--revision", default=BASE_REVISION)
    args = ap.parse_args()

    t0 = time.time()
    lm = HFLocalLM(revision=args.revision, batch_size=4)
    print(f"loaded on {lm.device} {lm.dtype} in {time.time() - t0:.0f}s; {len(lm.names)} tensors")
    print("model class:", type(lm.hf).__name__)
    print("sample names:", [n for n in lm.names if "layers.15." in n][:6])
    print("resolved names:", [lm.resolve(n) for n in NAMES])

    edits = build({n: lm.original(n) for n in NAMES})
    print("published edit rebuilt and sha256-verified against the evaluated checkpoint")

    dev, hold = load_gsm8k(n_dev=args.n, n_holdout=args.n)
    prog = MathSolver(lm, max_new_tokens=args.max_new_tokens)
    rows = {}
    t0 = time.time()
    rows["base (T=0.6)"] = score_rows(prog, hold, math_metric)
    print(f"base generation: {time.time() - t0:.0f}s for {args.n} problems")
    lm.apply_patch(edits)
    print("applied; changed entries:", lm.changed_entries())
    rows["published edit (T=0.6)"] = score_rows(prog, hold, math_metric)
    lm.reset()
    rows["base (T=1.0)"] = score_rows(prog, hold, math_metric, temperature=1.0)
    print(results_table(rows))


if __name__ == "__main__":
    main()
