"""Does the wrap-up signal predict accuracy? A check on the candidates of the six existing searches. Local, no GPU job.

For every accepted GEPA candidate (and the published edit) it measures how much more likely the edit makes a closing
word at a sentence end, and compares that with the candidate's held-out accuracy from candidate_curve.json and, for
the final edits, with MATH-500. Writes paper/wrapup_retro.md.

    python scripts/wrapup_retro.py --profile fe-vm-lakebase-praneeth
"""

from __future__ import annotations

import argparse
import json
import os

from analyze_directions import MAX_NEW, held_out_traces
from make_figures import CACHE, EVALS, SEARCHES, fetch
from scipy.stats import spearmanr

from dspy_weightspace import HFLocalLM
from dspy_weightspace.published_edit import BASE_REVISION
from dspy_weightspace.sandbox import run_fresh
from dspy_weightspace.wrapup import WrapUpProbe

FINAL_EVAL = {  # search label -> label of its MATH-500 eval in make_figures.EVALS
    "cold2": "cold2",
    "cold2 seed 1": "cold-s1",
    "cold2 seed 2": "cold-s2",
    "warm2": "warm2",
    "warm2 seed 1": "warm-s1",
    "warm2 seed 2": "warm-s2",
}
BASE = {"closers_x": 1.0, "starters_x": 1.0, "think_close_dlogp": 0.0, "kl_loop": 0.0, "kl_productive": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="paper/wrapup_retro.md")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    lm = HFLocalLM(revision=BASE_REVISION, max_tokens=MAX_NEW, batch_size=12)
    probe = WrapUpProbe(lm, [(t["problem"], t["text"]) for t in held_out_traces(lm)], max_tokens=MAX_NEW)
    print(lm.device, len(probe.rows), "traces", flush=True)

    seen: dict[str, dict] = {}  # identical patch scripts (the shared seeds) are measured once
    rows = []
    for label, run in SEARCHES.items():
        d = fetch(a.profile, run)
        cands = json.load(open(f"{d}/optimizer/gepa/candidates.json"))
        curve = {p["candidate"]: p for p in json.load(open(f"{d}/candidate_curve.json")) if "accuracy" in p}
        for i, cand in enumerate(cands):
            src = cand["weight_patch"]
            if src not in seen:
                if src.strip():
                    sb, res = run_fresh(lm, src, f"{d}/optimizer")
                    assert res.ok, (label, i, res.error)
                    sb.apply()
                    seen[src] = probe.measure()
                    lm.reset()
                else:
                    seen[src] = dict(BASE)
            final = i == max(curve, key=lambda k: curve[k]["selection"])
            rows.append(
                {
                    "run": label,
                    "candidate": i,
                    "parents": curve[i]["parents"],
                    "heldout": curve[i]["accuracy"],
                    "selection": curve[i]["selection"],
                    "math500": EVALS[FINAL_EVAL[label]][2] if final else None,
                    "first": src not in {r.get("_src") for r in rows},
                    "_src": src,
                    **seen[src],
                }
            )
            print(
                rows[-1]["run"],
                i,
                f"closers_x={rows[-1]['closers_x']:.2f}",
                f"heldout={rows[-1]['heldout']:.3f}",
                flush=True,
            )

    uniq = [r for r in rows if r["first"]]
    finals = [r for r in rows if r["math500"] is not None]
    rho_h = spearmanr([r["closers_x"] for r in uniq], [r["heldout"] for r in uniq])
    rho_m = spearmanr([r["closers_x"] for r in finals], [r["math500"] for r in finals])
    rho_s = spearmanr([r["selection"] for r in uniq], [r["heldout"] for r in uniq])
    by = {(r["run"], r["candidate"]): r for r in rows}
    steps = [(by[(r["run"], p)], r) for r in rows for p in (r["parents"] or []) if p is not None]
    agree = sum(
        (c["closers_x"] > p["closers_x"]) == (c["heldout"] > p["heldout"])
        for p, c in steps
        if c["heldout"] != p["heldout"]
    )
    moved = sum(c["heldout"] != p["heldout"] for p, c in steps)

    out = ["# Does the wrap-up signal predict accuracy? (existing searches)\n"]
    out.append(
        f"{len(uniq)} distinct candidates from six searches, {len(probe.rows)} base traces of {MAX_NEW:,} tokens. "
        "closers_x = how much more likely a closing word is at a sentence end, relative to the base model.\n"
    )
    out.append(
        f"- Rank correlation of closers_x with held-out accuracy (48 problems), all candidates: "
        f"{rho_h.statistic:.2f} (p={rho_h.pvalue:.3f})"
    )
    out.append(f"- Same for GEPA's selection score, for reference: {rho_s.statistic:.2f}")
    out.append(f"- Rank correlation of closers_x with MATH-500, six final edits: {rho_m.statistic:.2f}")
    out.append(
        f"- Parent to child steps where held-out accuracy changed: closers_x moved the same way in {agree} of {moved}\n"
    )
    heads = ["Run", "Candidate", "closers_x", "starters_x", "</think> dlogp", "KL productive", "Selection", "Held-out"]
    heads.append("MATH-500")
    out.append("| " + " | ".join(heads) + " |\n|" + "---|" * len(heads))
    for r in rows:
        m5 = f"{100 * r['math500']:.1f}%" if r["math500"] is not None else ""
        cells = [r["run"], r["candidate"], f"{r['closers_x']:.2f}", f"{r['starters_x']:.2f}"]
        cells += [f"{r['think_close_dlogp']:+.2f}", f"{r['kl_productive']:.3f}", f"{r['selection']:.3f}"]
        cells += [f"{100 * r['heldout']:.1f}%", m5]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    json.dump(
        [{k: v for k, v in r.items() if k != "_src"} for r in rows], open("/tmp/wrapup_retro.json", "w"), indent=1
    )
    print("\n".join(out[:8]))


if __name__ == "__main__":
    main()
