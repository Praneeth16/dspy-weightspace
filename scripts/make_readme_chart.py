"""The results chart: every method on MATH-500 with 95% bootstrap intervals, one point per search seed.

Reads the per-problem result files from the UC volume (cached in /tmp/figdata) through scripts/make_figures.py,
which stops if any accuracy differs from the paper's tables.

    python scripts/make_readme_chart.py --profile fe-vm-lakebase-praneeth --out docs/math500.png
    python scripts/make_readme_chart.py --profile fe-vm-lakebase-praneeth --portrait --out /tmp/math500_portrait.png

The portrait version (1080 x 1350) is for a phone feed: labels sit above their rows and the two methods a general
reader does not need (gradient training, difference of means) are left out.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from make_figures import AQUA, BLUE, CACHE, INK, MUTED, interval, per_problem  # noqa: E402

GRAY = "#6b6b66"
ORANGE = "#eb6834"
VIOLET = "#6250d6"
# (row label, portrait label, [(eval label, seed tag, chosen by held-out score)], colour)
OURS = [
    (
        "WeightSpaceOptimizer (GEPA), from scratch",
        "Search with GEPA",
        [("cold2", "seed 0", False), ("cold-s1", "seed 1", True), ("cold-s2", "seed 2", False)],
        AQUA,
    ),
    (
        "WeightSpaceOptimizer (GEPA), from the published edit",
        "Search with GEPA, started from the published edit",
        [("warm2", "seed 0", True), ("warm-s1", "seed 1", False), ("warm-s2", "seed 2", False)],
        BLUE,
    ),
    (
        "No GEPA loop: best of four independent proposals",
        "No GEPA: best of four one-shot scripts",
        [("nogepa-s0", "seed 0", True), ("nogepa-s1", "seed 1", False), ("nogepa-s2", "seed 2", False)],
        ORANGE,
    ),
    (
        "GEPA with a wrap-up probe for the proposer",
        "GEPA plus a tool that measures wrapping up",
        [("wrapup-s0", "seed 0", False), ("wrapup-s1", "seed 1", True), ("wrapup-s2", "seed 2", False)],
        VIOLET,
    ),
]
# (row label, portrait label or None to leave it out, eval label)
BASELINES = [
    ("Published edit (Astra)", "Published edit", "published"),
    ("Gradient training, same weights, human solutions", None, "ft-gold"),
    ("Base model, thinking off, no edit", "No edit, thinking switched off", "base-nothink35"),
    ("Difference-in-means direction, no LLM", None, "dm-C64-9"),
    ("Base model, thinking on", "No edit, thinking on", "base"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="docs/math500.png")
    ap.add_argument("--portrait", action="store_true")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tall = a.portrait
    base = 13.0 if tall else 10.5
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": base,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "ytick.left": False,
            "axes.edgecolor": MUTED,
            "xtick.color": MUTED,
            "savefig.pad_inches": 0.15,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 9.0) if tall else (9.6, 7.2))
    published = per_problem(a.profile, "published").mean() * 100
    ax.axvline(published, color=INK, lw=1, ls=(0, (4, 3)), zorder=1)
    white = {"fc": "white", "ec": "none", "pad": 1.5}  # keeps the reference line from crossing a label

    def point(y, label, color, filled, tag=""):
        acc = per_problem(a.profile, label)
        lo, hi = interval(acc)
        m = acc.mean() * 100
        ax.plot([lo * 100, hi * 100], [y, y], color=color, lw=2.6 if tall else 2.2, solid_capstyle="round", zorder=2)
        ax.plot(m, y, "o", ms=10 if tall else 8.5, mfc=color if filled else "white", mec=color, mew=2, zorder=3)
        bold = "bold" if filled and tag else "normal"
        ax.annotate(f"{m:.1f}", (hi * 100, y), xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=base, color=INK, fontweight=bold, bbox=white, zorder=4)  # fmt: skip
        if tag and not tall:
            ax.annotate(tag, (hi * 100, y), xytext=(42, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=MUTED, bbox=white, zorder=4)  # fmt: skip

    def heading(y, text):
        ax.annotate(text, (0, y), xytext=(0, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=base, color=INK, bbox=white, zorder=4)  # fmt: skip

    y, ticks, names = 0.0, [], []
    step = 0.62 if tall else 0.55
    for name, short, seeds, color in OURS:
        if tall:
            heading(y, short)
            y += 0.72
        ys = [y + step * i for i in range(len(seeds))]
        for yy, (label, tag, chosen) in zip(ys, seeds, strict=True):
            point(yy, label, color, chosen, tag)
        ticks.append(ys[1])
        names.append(name)
        y = ys[-1] + (1.05 if tall else 1.15)
    for name, short, label in BASELINES:
        if tall and short is None:
            continue
        if tall:
            heading(y, short)
            y += 0.72
        point(y, label, GRAY if label != "published" else INK, True)
        ticks.append(y)
        names.append(name)
        y += 1.0 if tall else 0.95

    if tall:
        ax.set_yticks([])
    else:
        ax.set_yticks(ticks, names, fontsize=base, color=INK)
        ax.tick_params(axis="y", length=0, pad=10)
    ax.set_ylim(y - 0.5, -1.05)
    ax.set_xlim(0, 72)
    ax.set_xticks(range(0, 71, 10), [f"{t}%" for t in range(0, 71, 10)])
    ax.xaxis.grid(True, color="#e9e8e3", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("MATH-500 accuracy", color=INK, labelpad=8)
    ax.annotate(f"published edit, {published:.1f}%", (published, ax.get_ylim()[1]), xytext=(5, -2),
                textcoords="offset points", va="top", fontsize=base - 1.5, color=INK)  # fmt: skip
    if tall:
        fig.suptitle("Same search, different random seed:\n32% to 61%", x=0.06, y=0.975, ha="left", va="top",
                     fontsize=21, fontweight="bold", color=INK, linespacing=1.25)  # fmt: skip
        fig.text(0.06, 0.872, "A larger model wrote weight edits for a small model (Qwen3.5-0.8B).\nEach dot is one "
                 "full search, scored on 500 math problems.\nLines are 95% intervals. Filled dots are the runs a "
                 "held-out set picks.", ha="left", va="top", fontsize=11.5, color=MUTED, linespacing=1.5)  # fmt: skip
        fig.subplots_adjust(top=0.775, bottom=0.075, left=0.06, right=0.94)
        fig.savefig(a.out, dpi=150, facecolor="white")
    else:
        fig.suptitle("A search over weight patches beats the published edit, but only in some runs", x=0.012,
                     y=0.995, ha="left", fontsize=13.5, fontweight="bold", color=INK)  # fmt: skip
        fig.text(0.012, 0.945, "Qwen3.5-0.8B on MATH-500: 500 problems, 2 samples each, strict grading. Lines are 95% "
                 "bootstrap intervals over problems.\nEach search ran with three seeds. Filled points mark the run a "
                 "48-problem held-out set would choose.", ha="left", va="top", fontsize=9.5, color=MUTED,
                 linespacing=1.5)  # fmt: skip
        fig.subplots_adjust(top=0.85, left=0.4, right=0.985)
        fig.savefig(a.out, dpi=220, facecolor="white", bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
