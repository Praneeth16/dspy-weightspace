"""Score MATH-500 results in two groups: problems the base model solved, and problems where it only looped.

A reader's check on wrap-up edits: do they fix the loops without rushing the problems the model already solved?

    python scripts/group_scores.py --profile fe-vm-lakebase-praneeth [label ...]      (labels from make_figures.EVALS)
"""

from __future__ import annotations

import argparse
import json
import os

from make_figures import CACHE, EVALS, fetch

from dspy_weightspace.baseline import group_scores

DEFAULT = ["base", "base-nothink35", "published", "cold-s1", "warm-s1", "warm2", "cold-s2", "cold2", "warm-s2"]


def rows(profile: str, label: str) -> list[dict]:
    run, stem, _ = EVALS[label]
    return [json.loads(x) for x in open(f"{fetch(profile, run)}/{stem}.jsonl")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("labels", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    base = rows(a.profile, "base")
    heads = ["Variant", "Overall", "Base solved", "Tokens when correct", "Base looped", "Cut at limit"]
    print("| " + " | ".join(heads) + " |\n|" + "---|" * len(heads))
    for label in a.labels:
        r = rows(a.profile, label)
        g = group_scores(base, r)
        s, lo = g["base_solved"], g["base_looped"]
        cells = [label, f"{sum(x['correct'] for x in r) / len(r):.1%}", f"{s['accuracy']:.1%} (n={s['problems']})"]
        cells += [f"{s['tokens_when_correct'] / 1000:.1f}k", f"{lo['accuracy']:.1%} (n={lo['problems']})"]
        cells.append(f"{lo['cut_rate']:.1%}")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
