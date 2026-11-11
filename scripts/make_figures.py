"""Paper figures F1 to F4 from the run files in the UC volume. Local (CPU), no GPU job.

F1 angle between each edit's direction and the published one, F2 selection score vs held-out accuracy per GEPA
candidate, F3 the MATH-500 direction x scope x strength sweep, F4 lenient vs strict grading. Writes PDFs (and PNG
previews) to --out. Each plotted MATH-500 accuracy is recomputed from the per-problem jsonl and checked against the
paper's tables; the script stops on any mismatch.

    python scripts/make_figures.py --profile fe-vm-lakebase-praneeth --out paper/latex/figures
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dspy_weightspace import HFLocalLM  # noqa: E402
from dspy_weightspace.published_edit import BASE_REVISION, build  # noqa: E402
from dspy_weightspace.sandbox import run_fresh  # noqa: E402

VOL = "dbfs:/Volumes/serverless_lakebase_praneeth_catalog/weightspace/artifacts"
CACHE = "/tmp/figdata"
LAYERS = (15, 19)
SWEEP = "runs/20261108-233758-sweep"
# label: (eval run dir, jsonl stem, accuracy in the paper's tables)
EVALS = {
    "base": ("runs/20261108-120505-vllm", "base", 0.071),
    "published": ("runs/20261108-120505-vllm", "published", 0.533),
    "pub-all-6": ("runs/20261108-181021-vllm", "pub-all-6", 0.464),
    "dm-C64-9": ("runs/20261108-181021-vllm", "dm-C64-9", 0.298),
    "dm-C64-6": ("runs/20261108-181021-vllm", "dm-C64-6", 0.271),
    "dm-all-9": ("runs/20261108-181021-vllm", "dm-all-9", 0.211),
    "cold-C64-6": ("runs/20261108-181032-vllm", "cold-C64-6", 0.182),
    "rand-C64-6": ("runs/20261108-181032-vllm", "rand-C64-6", 0.061),
    "base-nothink35": ("runs/20261108-120505-vllm", "base-nothink35", 0.456),
    "ft-gold": ("runs/20261108-152338-vllm", "ft-gold", 0.476),
    "cold2": ("runs/20261108-213344-f3e6f0-vllm", "cold2", 0.320),
    "cold-s1": ("runs/20261109-183108-dc9807-vllm", "cold-s1", 0.606),
    "cold-s2": ("runs/20261109-183123-247aa7-vllm", "cold-s2", 0.406),
    "warm2": ("runs/20261108-213344-2ad745-vllm", "warm2", 0.543),
    "warm-s1": ("runs/20261109-182657-ebefd6-vllm", "warm-s1", 0.599),
    "warm-s2": ("runs/20261109-184102-fbf676-vllm", "warm-s2", 0.325),
}
# label: search run dir. Repeat runs are added here once they finish.
SEARCHES = {
    "cold2": "runs/20261108-202105-00c880",
    "warm2": "runs/20261108-202106-03139f",
    "cold2 seed 1": "runs/20261109-152059-31c009",
    "warm2 seed 1": "runs/20261109-152059-28844d",
    "cold2 seed 2": "runs/20261109-153128-3036f2",
    "warm2 seed 2": "runs/20261109-152628-b52fb7",
}
# label: run dir whose patch.py (with artifacts in optimizer/) rebuilds the edit
EDITS = {
    "difference of means": f"{SWEEP}/dm-C64-9",
    "cold search, GSM8K selection": f"{SWEEP}/cold-C64-6",
    "random": f"{SWEEP}/rand-C64-6",
    # Warm runs keep the published edit on layers 15 and 19 (cosine 1), so only cold searches are shown.
    **{k: v for k, v in SEARCHES.items() if k.startswith("cold")},
}
# Lenient grading counted a boxed answer inside unfinished thinking. Only these two runs (vLLM 0.24, AIR runs
# 417912843168672 and 25685828757048) were graded both ways; later jsonl files keep only the strict answer.
GRADING = {"base": (0.216, 0.080), "published": (0.561, 0.519)}

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85"
INK, MUTED = "#1f1f1e", "#5f5e5a"
COL, FULL = 3.25, 6.75  # ICML column and text widths, inches


def fetch(profile: str, src: str) -> str:
    dest = f"{CACHE}/{src.replace('/', '_')}"
    if not os.path.exists(dest):
        cmd = ["databricks", "fs", "cp", "-r", f"{VOL}/{src}", dest, "--overwrite", "--profile", profile]
        subprocess.run(cmd, check=True, capture_output=True)
    return dest


def per_problem(profile: str, label: str) -> np.ndarray:
    """Mean correctness per MATH-500 problem (2 samples each), indexed by problem."""
    run, stem, expected = EVALS[label]
    rows = [json.loads(x) for x in open(f"{fetch(profile, run)}/{stem}.jsonl")]
    by = {}
    for r in rows:
        by.setdefault(r["idx"], []).append(float(r["correct"]))
    assert len(by) == 500 and all(len(v) == 2 for v in by.values()), (label, len(by))
    acc = np.array([np.mean(by[i]) for i in sorted(by)])
    assert abs(acc.mean() - expected) < 5e-4, (label, acc.mean(), expected)
    return acc


def interval(x: np.ndarray, n: int = 10_000) -> tuple[float, float]:
    """95% bootstrap interval of the mean, resampling problems."""
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save(fig, out: str, name: str) -> None:
    fig.savefig(f"{out}/{name}.pdf")
    fig.savefig(f"{out}/{name}.png", dpi=200)
    plt.close(fig)


def edit_directions(profile: str) -> dict[str, dict[int, np.ndarray]]:
    """Top left singular vector of each edit's weight change, per layer. For W - a u u^T W this is u itself."""
    lm = HFLocalLM(revision=BASE_REVISION, device="cpu", dtype=torch.bfloat16)
    hf_names = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in LAYERS]
    names = [lm.resolve(n) for n in hf_names]
    published = build({n: lm.original(n) for n in hf_names}, fetch(profile, "exp8_edit"))
    edits = {"published": {lm.resolve(n): t for n, t in published.items()}}
    for label, run in EDITS.items():
        d = fetch(profile, run)
        sb, res = run_fresh(lm, open(f"{d}/patch.py").read(), f"{d}/optimizer")
        assert res.ok, (label, res.error)
        edits[label] = dict(sb.edits)
    out = {}
    for label, e in edits.items():
        out[label] = {}
        for L, n in zip(LAYERS, names, strict=True):
            if n in e:
                dw = (e[n].double() - lm.original(n).double()).numpy()
                out[label][L] = np.linalg.svd(dw, full_matrices=False)[0][:, 0]
    return out


def f1(profile: str, out: str) -> dict:
    dirs = edit_directions(profile)
    labels = [k for k in dirs if k != "published"]
    cos = {
        k: {L: float(abs(dirs[k][L] @ dirs["published"][L])) if L in dirs[k] else None for L in LAYERS} for k in labels
    }
    rng = np.random.default_rng(0)
    pairs = rng.standard_normal((2000, 2, 1024))
    chance = float(np.mean([abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in pairs]))
    fig, ax = plt.subplots(figsize=(COL, 0.32 * len(labels) + 0.55))
    y = np.arange(len(labels))[::-1]
    for L, marker, color, dy in ((15, "o", BLUE, 0.12), (19, "s", ORANGE, -0.12)):
        xs = [cos[k][L] if cos[k][L] is not None else np.nan for k in labels]
        ax.scatter(xs, y + dy, marker=marker, s=22, color=color, edgecolors="white", linewidths=0.8, label=f"layer {L}")
        for x, yy in zip(xs, y + dy, strict=True):
            if not np.isnan(x):
                ax.annotate(
                    f"{x:.2f}", (x, yy), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.5, color=INK
                )
    ax.axvline(chance, color=GRAY, lw=1, ls="--")
    ax.annotate(
        "two random\ndirections",
        (chance, y[-1] - 0.45),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=6.5,
        color=MUTED,
        va="center",
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(y[-1] - 0.75, y[0] + 0.5)
    ax.set_xlabel("|cosine| with the published direction")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.2)
    save(fig, out, "f1_direction_cosines")
    return {"cosine_with_published": cos, "chance": chance}


def f2(profile: str, out: str) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), sharex=True, sharey=True)
    curves = {}
    markers = {"0": "o", "1": "^", "2": "s"}
    for label, run in SEARCHES.items():
        c = json.load(open(f"{fetch(profile, run)}/candidate_curve.json"))
        c = [p for p in c if "selection" in p]
        curves[label] = [(p["selection"], p["accuracy"]) for p in c]
        warm = label.startswith("warm")
        seed = label.rsplit(" ", 1)[-1] if "seed" in label else "0"
        ax, color, marker = axes[int(warm)], BLUE if warm else AQUA, markers[seed]
        by = {p["candidate"]: p for p in c}
        for p in c:
            for parent in p["parents"] or []:
                if parent is not None and parent in by:
                    q = by[parent]
                    ax.annotate(
                        "",
                        (p["selection"], p["accuracy"]),
                        (q["selection"], q["accuracy"]),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.9, "shrinkA": 4, "shrinkB": 4},
                    )
        ax.scatter(
            [p["selection"] for p in c],
            [p["accuracy"] for p in c],
            s=24,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        last = max(c, key=lambda p: p["selection"])
        dy = -9 if (warm and seed == "1") else 2
        ax.annotate(
            f"seed {seed}",
            (last["selection"], last["accuracy"]),
            xytext=(5, dy),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
    for ax, title in zip(axes, ("Cold start", "Warm start (from the published edit)"), strict=True):
        ax.set_title(title, color=INK)
        ax.set_xlabel("selection score (32 MATH-train problems)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("held-out accuracy (48 problems)")
    axes[1].set_xlim(right=axes[1].get_xlim()[1] + 0.03)
    save(fig, out, "f2_candidate_curves")
    return curves


def f3(profile: str, out: str) -> dict:
    order = ["published", "pub-all-6", "dm-C64-9", "dm-C64-6", "dm-all-9", "cold-C64-6", "rand-C64-6", "base"]
    colors = {"pub": BLUE, "dm": ORANGE, "cold": AQUA, "rand": GRAY, "base": GRAY}
    res = {}
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    for i, k in enumerate(order):
        a = per_problem(profile, k)
        lo, hi = interval(a)
        res[k] = (float(a.mean()), lo, hi)
        fam = "pub" if k == "published" else k.split("-")[0]
        ax.bar(
            i,
            a.mean(),
            width=0.62,
            color=colors[fam] if k != "base" else "white",
            edgecolor=colors[fam],
            linewidth=1,
        )
        ax.errorbar(i, a.mean(), yerr=[[a.mean() - lo], [hi - a.mean()]], color=INK, lw=0.8, capsize=2)
        ax.annotate(
            f"{a.mean():.1%}", (i, hi), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK
        )
    names = {
        "published": "published\n64 cols\nα=6",
        "pub-all-6": "published\nall cols\nα=6",
        "dm-C64-9": "diff. means\n64 cols\nα=9",
        "dm-C64-6": "diff. means\n64 cols\nα=6",
        "dm-all-9": "diff. means\nall cols\nα=9",
        "cold-C64-6": "cold search\n64 cols\nα=6",
        "rand-C64-6": "random\n64 cols\nα=6",
        "base": "base\nno edit\n",
    }
    ax.set_xticks(range(len(order)), [names[k] for k in order], fontsize=6.5)
    ax.set_ylabel("MATH-500 accuracy")
    ax.set_ylim(0, 0.66)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    save(fig, out, "f3_sweep")
    return res


def f4(out: str) -> dict:
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    x = np.arange(len(GRADING))
    for j, (name, color) in enumerate((("lenient", "white"), ("strict", BLUE))):
        vals = [GRADING[k][j] for k in GRADING]
        ax.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=color, edgecolor=BLUE, linewidth=1, label=name)
        for xi, v in zip(x, vals, strict=True):
            ax.annotate(
                f"{v:.1%}",
                (xi + (j - 0.5) * 0.36, v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=INK,
            )
    ax.set_xticks(x, list(GRADING))
    ax.set_ylabel("MATH-500 accuracy")
    ax.set_ylim(0, 0.68)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    save(fig, out, "f4_grading")
    return GRADING


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="paper/latex/figures")
    ap.add_argument("--only", default="f1,f2,f3,f4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    style()
    figs = {
        "f1": lambda: f1(a.profile, a.out),
        "f2": lambda: f2(a.profile, a.out),
        "f3": lambda: f3(a.profile, a.out),
        "f4": lambda: f4(a.out),
    }
    numbers = {k: figs[k]() for k in a.only.split(",")}
    with open(f"{a.out}/numbers.json", "w") as f:
        json.dump(numbers, f, indent=2, default=str)
    print(json.dumps(numbers, indent=2, default=str))


if __name__ == "__main__":
    main()
