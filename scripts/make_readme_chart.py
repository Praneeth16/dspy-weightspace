"""The README's results chart: every method on MATH-500 with 95% bootstrap intervals, one point per search seed.

Reads the per-problem result files from the UC volume (cached in /tmp/figdata) through scripts/make_figures.py,
which stops if any accuracy differs from the paper's tables.

    python scripts/make_readme_chart.py --profile fe-vm-lakebase-praneeth --out docs/math500.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from make_figures import AQUA, BLUE, CACHE, INK, MUTED, interval, per_problem  # noqa: E402

GRAY = "#6b6b66"
# (row label, [(eval label, seed tag, chosen by held-out score)], colour)
OURS = [
    (
        "WeightSpaceOptimizer, from scratch",
        [("cold2", "seed 0", False), ("cold-s1", "seed 1", True), ("cold-s2", "seed 2", False)],
        AQUA,
    ),
    (
        "WeightSpaceOptimizer, from the published edit",
        [("warm2", "seed 0", True), ("warm-s1", "seed 1", False), ("warm-s2", "seed 2", False)],
        BLUE,
    ),
]
BASELINES = [
    ("Published edit (Astra)", "published"),
    ("Gradient training, same weights, human solutions", "ft-gold"),
    ("Base model, thinking off, no edit", "base-nothink35"),
    ("Difference-in-means direction, no LLM", "dm-C64-9"),
    ("Base model, thinking on", "base"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="docs/math500.png")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "ytick.left": False,
            "axes.edgecolor": MUTED,
            "xtick.color": MUTED,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    published = per_problem(a.profile, "published").mean() * 100
    ax.axvline(published, color=INK, lw=1, ls=(0, (4, 3)), zorder=1)

    def point(y, label, color, filled, tag=""):
        acc = per_problem(a.profile, label)
        lo, hi = interval(acc)
        m = acc.mean() * 100
        ax.plot([lo * 100, hi * 100], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot(m, y, "o", ms=8.5, mfc=color if filled else "white", mec=color, mew=2, zorder=3)
        white = {"fc": "white", "ec": "none", "pad": 1.5}  # keeps the reference line from crossing a label
        ax.annotate(
            f"{m:.1f}",
            (hi * 100, y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=10.5,
            color=INK,
            fontweight="bold" if filled and tag else "normal",
            bbox=white,
            zorder=4,
        )
        if tag:
            ax.annotate(
                tag,
                (hi * 100, y),
                xytext=(42, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=MUTED,
                bbox=white,
                zorder=4,
            )

    y, ticks, names = 0.0, [], []
    for name, seeds, color in OURS:
        ys = [y + 0.55 * i for i in range(len(seeds))]
        for yy, (label, tag, chosen) in zip(ys, seeds, strict=True):
            point(yy, label, color, chosen, tag)
        ticks.append(ys[1])
        names.append(name)
        y = ys[-1] + 1.15
    for name, label in BASELINES:
        point(y, label, GRAY if label != "published" else INK, True)
        ticks.append(y)
        names.append(name)
        y += 0.95

    ax.set_yticks(ticks, names, fontsize=10.5, color=INK)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_ylim(y - 0.5, -1.05)
    ax.set_xlim(0, 72)
    ax.set_xticks(range(0, 71, 10), [f"{t}%" for t in range(0, 71, 10)])
    ax.xaxis.grid(True, color="#e9e8e3", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("MATH-500 accuracy", color=INK, labelpad=8)
    ax.annotate(
        f"published edit, {published:.1f}%",
        (published, ax.get_ylim()[1]),
        xytext=(5, -2),
        textcoords="offset points",
        va="top",
        fontsize=9,
        color=INK,
    )
    fig.suptitle(
        "A search over weight patches beats the published edit, but only in some runs",
        x=0.012,
        y=0.995,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.012,
        0.925,
        "Qwen3.5-0.8B on MATH-500: 500 problems, 2 samples each, strict grading. Lines are 95% bootstrap "
        "intervals over problems.\nEach search ran with three seeds. Filled points mark the run a 48-problem held-out "
        "set would choose.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=MUTED,
        linespacing=1.5,
    )
    fig.subplots_adjust(top=0.8, left=0.37, right=0.985)
    fig.savefig(a.out, dpi=220, facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
