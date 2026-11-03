"""Print tails and distinct-8gram ratios of base vs published-edit generations to see if cut-offs are loops."""

import sys
import time

from dspy_weightspace import HFLocalLM, MathSolver, load_gsm8k
from dspy_weightspace.published_edit import BASE_REVISION, build

n, max_new = int(sys.argv[1]), int(sys.argv[2])
lm = HFLocalLM(revision=BASE_REVISION, batch_size=n)
prog = MathSolver(lm, max_new_tokens=max_new)
_, hold = load_gsm8k(n_dev=n, n_holdout=n)
NAMES = ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"]


def ngram_ratio(text, k=8, tail=600):
    toks = text.split()[-tail:]
    grams = [tuple(toks[i : i + k]) for i in range(max(0, len(toks) - k))]
    return len(set(grams)) / max(1, len(grams))


for label, patch in [("base", None), ("published", build({m: lm.original(m) for m in NAMES}))]:
    lm.reset()
    if patch:
        lm.apply_patch(patch)
    t0 = time.time()
    preds = prog.batch_forward([e.problem for e in hold], seed=0)
    print(f"\n===== {label}: {time.time() - t0:.0f}s, {sum(p.n_tokens for p in preds)} tokens")
    for e, p in zip(hold, preds, strict=True):
        r = ngram_ratio(p.text)
        has_box = "\\boxed" in p.text
        print(f"--- gold={e.answer} finish={p.finish_reason} n={p.n_tokens} distinct8gram={r:.2f} boxed={has_box}")
        print("   HEAD:", p.text[:200].replace("\n", " "))
        print("   TAIL:", p.text[-300:].replace("\n", " "))
