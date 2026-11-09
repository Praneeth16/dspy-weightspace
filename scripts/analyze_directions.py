"""What does the published direction read and what does its edit suppress? Compared with the difference-in-means
direction and the cold-start optimizer edit (cold2), on base-model traces. Local (MPS or CPU), no GPU job.

Writes paper/direction_analysis.md and /tmp/direction_analysis.json. Traces are cached in /tmp/analysis_traces.json.

    python scripts/analyze_directions.py --profile fe-vm-lakebase-praneeth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import torch

from dspy_weightspace import HFLocalLM, load_math
from dspy_weightspace.data import to_chat
from dspy_weightspace.metrics import LOOP_THRESHOLD, repetition
from dspy_weightspace.published_edit import BASE_REVISION, build
from dspy_weightspace.sandbox import run_fresh

VOL = "dbfs:/Volumes/serverless_lakebase_praneeth_catalog/weightspace/artifacts"
RUNS = {
    "dm-C64-9": "runs/20261108-233758-sweep/dm-C64-9",
    "cold2": "runs/20261108-202105-00c880",
}
DM_DIRS = "runs/20261108-124746-diffmean"
LAYERS = (15, 19)
STARTERS = [" Wait", "Wait", " But", " Alternatively", " Hmm", " However", " Let", " Actually"]
CLOSERS = ["</think>", " So", " Therefore", " Thus", " Final", "**"]
# Base traces start looping only after a few thousand tokens; 1,536 had none.
N_TRACES, MAX_NEW, CHUNK = 12, 6144, 128


def fetch(profile: str, src: str, dest: str) -> str:
    cmd = ["databricks", "fs", "cp", "-r", f"{VOL}/{src}", dest, "--overwrite", "--profile", profile]
    subprocess.run(cmd, check=True, capture_output=True)
    return dest


def single_ids(tok, words):
    out = {}
    for w in words:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            out[w] = ids[0]
    return out


def loop_labels(tok, ids: list[int]) -> np.ndarray:
    """Per completion token: inside a looping stretch? Chunks of CHUNK tokens, judged on the preceding 1,024 tokens."""
    lab = np.zeros(len(ids), dtype=bool)
    for s in range(0, len(ids), CHUNK):
        window = tok.decode(ids[max(0, s - 1024) : s + CHUNK])
        lab[s : s + CHUNK] = repetition(window)[0] < LOOP_THRESHOLD
    return lab


def effect(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-9
    return float((a.mean() - b.mean()) / pooled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="paper/direction_analysis.md")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    lm = HFLocalLM(revision=BASE_REVISION, max_tokens=MAX_NEW, batch_size=12)
    tok = lm.tok
    layers = lm._layers()
    print(lm.device, lm.dtype, flush=True)

    # ----- edits and directions -------------------------------------------------------------------------------
    names = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in LAYERS]
    pub = {
        lm.resolve(n): t
        for n, t in build({n: lm.original(n) for n in names}, fetch(a.profile, "exp8_edit", f"{tmp}/e")).items()
    }
    edits = {"published": pub}
    for label, run in RUNS.items():
        d = fetch(a.profile, run, f"{tmp}/{label}")
        sb, res = run_fresh(lm, open(f"{d}/patch.py").read(), f"{d}/optimizer")
        assert res.ok, (label, res.error)
        edits[label] = dict(sb.edits)
    fetch(a.profile, DM_DIRS, f"{tmp}/dm")
    dirs, cols = {"pub": {}, "dm": {}, "cold2": {}}, {}
    for L in LAYERS:
        n = lm.resolve(f"model.layers.{L}.mlp.down_proj.weight")
        dw = (pub[n] - lm.original(n)).double()
        c = (dw != 0).any(0).nonzero().ravel()
        cols[L] = c
        dirs["pub"][L] = torch.linalg.svd(dw[:, c], full_matrices=False).U[:, 0].float()
        dirs["dm"][L] = torch.from_numpy(np.load(f"{tmp}/dm/dirs_L{L + 1}.npy")).float()
        dirs["cold2"][L] = (
            torch.from_numpy(np.load(f"{tmp}/cold2/optimizer/artifacts/loopdir_L{L}.npy")).float().ravel()
        )
        for k in dirs:
            dirs[k][L] = dirs[k][L] / dirs[k][L].norm()

    # ----- traces --------------------------------------------------------------------------------------------
    cache = f"/tmp/analysis_traces_{MAX_NEW}.json"
    if os.path.exists(cache):
        traces = json.load(open(cache))
    else:
        dev, hold = load_math(n_dev=48, n_holdout=48, seed=0)
        used = {e.problem for e in dev + hold}
        pool, _ = load_math(n_dev=200, n_holdout=0, seed=7)
        probs = [e.problem for e in pool if e.problem not in used][:N_TRACES]
        comps = lm.generate_detailed([to_chat(p) for p in probs], max_new_tokens=MAX_NEW, temperature=0.6, seed=0)
        traces = [{"problem": p, "text": c.text} for p, c in zip(probs, comps, strict=True)]
        json.dump(traces, open(cache, "w"))
    print(f"{len(traces)} traces", flush=True)

    rows = []  # per trace: prompt length, completion ids, loop labels, boundary flags
    for t in traces:
        p_ids = tok(lm._render(to_chat(t["problem"])), add_special_tokens=False)["input_ids"]
        c_ids = tok(t["text"], add_special_tokens=False)["input_ids"][:MAX_NEW]
        prev = [tok.decode([i]) for i in c_ids]
        boundary = np.array(
            [k > 0 and prev[k - 1].rstrip(" ").endswith((".", "\n", "?", ":", "!")) for k in range(len(c_ids))]
        )
        rows.append({"p": p_ids, "c": c_ids, "loop": loop_labels(tok, c_ids), "boundary": boundary})
    loop_all = np.concatenate([r["loop"] for r in rows])
    print(f"tokens {len(loop_all)}, looping {loop_all.mean():.0%}", flush=True)

    # ----- 1. where u is written: MLP output projections and per-neuron writers (base weights) -------------------
    captured = {}

    def hook(L):
        def fn(mod, inp, out):
            captured[L] = (inp[0][0].float().cpu(), out[0].float().cpu())

        return fn

    hs = [layers[L].mlp.down_proj.register_forward_hook(hook(L)) for L in LAYERS]
    proj = {k: {L: [] for L in LAYERS} for k in dirs}
    neuron = {L: [] for L in LAYERS}
    diffvec = {L: [torch.zeros(1024), torch.zeros(1024), 0, 0] for L in LAYERS}
    sanity = {}
    with torch.no_grad():
        for r in rows:
            ids = torch.tensor([r["p"] + r["c"]], device=lm.device)
            lm.hf.model(input_ids=ids)  # decoder only: full-vocabulary logits for 6k tokens would not fit
            n_p = len(r["p"])
            for L in LAYERS:
                x, y = captured[L][0][n_p:], captured[L][1][n_p:]
                for k in dirs:
                    proj[k][L].append((y @ dirs[k][L]).numpy())
                n = lm.resolve(f"model.layers.{L}.mlp.down_proj.weight")
                wu = dirs["pub"][L] @ lm.original(n)[:, cols[L]]  # u^T W[:, j] for the 64 columns
                neuron[L].append((x[:, cols[L]] * wu).numpy())
                lp, pp = torch.from_numpy(r["loop"]), ~torch.from_numpy(r["loop"])
                diffvec[L][0] += y[lp].sum(0)
                diffvec[L][1] += y[pp].sum(0)
                diffvec[L][2] += int(lp.sum())
                diffvec[L][3] += int(pp.sum())
                if L not in sanity:  # the edit's change to the u-projection must equal -6 x the 64 columns' share
                    dw = pub[n] - lm.original(n)
                    change = (x @ dw.T) @ dirs["pub"][L]
                    share = x[:, cols[L]] @ (lm.original(n)[:, cols[L]].T @ dirs["pub"][L])
                    sanity[L] = float((change + 6 * share).abs().max() / (share.abs().max() + 1e-9))
    for h in hs:
        h.remove()

    write = {}
    for k in dirs:
        write[k] = {}
        for L in LAYERS:
            v = np.concatenate(proj[k][L])
            write[k][L] = {
                "loop": float(v[loop_all].mean()),
                "productive": float(v[~loop_all].mean()),
                "d": effect(v[loop_all], v[~loop_all]),
            }
    geometry = {}
    for L in LAYERS:
        d = diffvec[L][0] / max(1, diffvec[L][2]) - diffvec[L][1] / max(1, diffvec[L][3])
        d = d / d.norm()
        geometry[L] = {k: float(abs(d @ dirs[k][L])) for k in dirs}
        geometry[L].update(
            {
                f"{x}·{y}": float(abs(dirs[x][L] @ dirs[y][L]))
                for x, y in (("pub", "dm"), ("pub", "cold2"), ("dm", "cold2"))
            }
        )

    contexts = {}
    for L in LAYERS:
        flat = [(float(v), ti, pos) for ti, arr in enumerate(proj["pub"][L]) for pos, v in enumerate(arr)]
        flat.sort(reverse=True)
        contexts[L] = []
        for v, ti, pos in flat[:15]:
            c = rows[ti]["c"]
            contexts[L].append(
                {
                    "proj": v,
                    "loop": bool(rows[ti]["loop"][pos]),
                    "text": tok.decode(c[max(0, pos - 10) : pos])
                    + " [["
                    + tok.decode(c[pos : pos + 1])
                    + "]] "
                    + tok.decode(c[pos + 1 : pos + 4]),
                }
            )
    neurons = {}
    for L in LAYERS:
        m = np.concatenate(neuron[L])  # [tokens, 64]: each column's write onto u
        gap = m[loop_all].mean(0) - m[~loop_all].mean(0)
        top = np.argsort(-np.abs(gap))[:5]
        neurons[L] = [
            {
                "column": int(cols[L][j]),
                "loop_minus_productive": float(gap[j]),
                "mean_abs": float(np.abs(m[:, j]).mean()),
            }
            for j in top
        ]
        neurons[L].append({"share_of_gap_top5": float(np.abs(gap[top]).sum() / (np.abs(gap).sum() + 1e-9))})

    # ----- 2. what each edit changes in next-token predictions (causal) ------------------------------------------
    starters, closers = single_ids(tok, STARTERS), single_ids(tok, CLOSERS)
    groups = {"starters": list(starters.values()), "closers": list(closers.values())}
    single = {**{f"S:{w}": i for w, i in starters.items()}, **{f"C:{w}": i for w, i in closers.items()}}

    head = lm.hf.get_output_embeddings()

    def final_hidden(ids):
        with torch.no_grad():
            return lm.hf.model(input_ids=ids).last_hidden_state[0]

    causal = {
        k: {
            "kl_loop": [],
            "kl_prod": [],
            **{f"{g}_{pt}": [] for g in groups for pt in ("boundary", "other")},
            # probability mass of each group (not log): log-prob means are dominated by positions where the token is
            # far out of reach, so they overstate shifts that change little probability.
            **{f"{g}_p_{pt}": [] for g in groups for pt in ("boundary", "other")},
            **{f"{s}_boundary": [] for s in single},
        }
        for k in ["base", *edits]
    }
    for r in rows:
        ids = torch.tensor([r["p"] + r["c"]], device=lm.device)
        n_p = len(r["p"])
        sl = slice(n_p - 1, n_p - 1 + len(r["c"]))  # logits that predict each completion token
        hb = final_hidden(ids)[sl]
        for k in ["base", *edits]:
            if k != "base":
                lm.apply_patch(edits[k])
            he = hb if k == "base" else final_hidden(ids)[sl]
            if k != "base":
                lm.reset()
                assert lm.changed_entries() == 0
            kl, dg, ds, pg = [], {g: [] for g in groups}, {x: [] for x in single}, {g: [] for g in groups}
            for c0 in range(0, hb.shape[0], 512):  # full-vocabulary log-probs, 512 positions at a time
                with torch.no_grad():
                    lb = torch.log_softmax(head(hb[c0 : c0 + 512]).float(), -1)
                    le = lb if k == "base" else torch.log_softmax(head(he[c0 : c0 + 512]).float(), -1)
                    kl.append((lb.exp() * (lb - le)).sum(-1).cpu())
                    for g, gids in groups.items():
                        dg[g].append((torch.logsumexp(le[:, gids], -1) - torch.logsumexp(lb[:, gids], -1)).cpu())
                        pg[g].append(torch.logsumexp(le[:, gids], -1).exp().cpu())
                    for x, i in single.items():
                        ds[x].append((le[:, i] - lb[:, i]).cpu())
            kl = torch.cat(kl).numpy()
            causal[k]["kl_loop"].append(kl[r["loop"]])
            causal[k]["kl_prod"].append(kl[~r["loop"]])
            for g in groups:
                d = torch.cat(dg[g]).numpy()
                causal[k][f"{g}_boundary"].append(d[r["boundary"]])
                causal[k][f"{g}_other"].append(d[~r["boundary"]])
                m = torch.cat(pg[g]).numpy()
                causal[k][f"{g}_p_boundary"].append(m[r["boundary"]])
                causal[k][f"{g}_p_other"].append(m[~r["boundary"]])
            for x in single:
                causal[k][f"{x}_boundary"].append(torch.cat(ds[x]).numpy()[r["boundary"]])
    causal = {k: {m: float(np.concatenate(v).mean()) for m, v in d.items()} for k, d in causal.items()}

    # ----- 3. logit lens (weak evidence at mid layers) --------------------------------------------------------
    head_w = head.weight.float().cpu()
    norm = lm.hf.model.norm.weight.float().cpu() if hasattr(lm.hf.model, "norm") else torch.ones(head_w.shape[1])
    lens = {}
    for k in dirs:
        lens[k] = {}
        for L in LAYERS:
            s = head_w @ (norm * dirs[k][L])
            lens[k][L] = {
                "+": [tok.decode([i]) for i in s.topk(12).indices.tolist()],
                "-": [tok.decode([i]) for i in (-s).topk(12).indices.tolist()],
            }

    out = {
        "tokens": int(len(loop_all)),
        "looping_share": float(loop_all.mean()),
        "sanity_edit_identity_relerr": sanity,
        "write": write,
        "geometry": geometry,
        "contexts": contexts,
        "neurons": neurons,
        "causal": causal,
        "logit_lens": lens,
    }
    json.dump(out, open("/tmp/direction_analysis.json", "w"), indent=2, default=str)
    write_report(out, a.out)
    print("wrote", a.out, flush=True)


def row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def write_report(o: dict, path: str) -> None:
    L = []
    L.append("# What the published direction reads and what its edit suppresses\n")
    L.append(
        f"{o['tokens']:,} completion tokens from {N_TRACES} base traces on held-out MATH-train problems "
        f"({MAX_NEW:,} tokens each, T=0.6); {o['looping_share']:.0%} of tokens fall in looping stretches. "
        f"Sanity: the edit's change to the u-projection equals −6 × the 64 columns' share "
        f"(max relative error {max(o['sanity_edit_identity_relerr'].values()):.1e}).\n"
    )
    L.append("## 1. How sharply each direction is written on looping vs productive tokens (base MLP output)\n")
    L.append("| Direction | Layer | Mean on looping | Mean on productive | Effect size d |\n|---|---|---|---|---|")
    for k, by in o["write"].items():
        for lay, v in by.items():
            L.append(f"| {k} | {lay} | {v['loop']:+.3f} | {v['productive']:+.3f} | {v['d']:+.2f} |")
    L.append(
        "\n## 2. Alignment with the looping-minus-productive mean of the MLP output, and with each other (|cos|)\n"
    )
    L.append("| Layer | pub | dm | cold2 | pub·dm | pub·cold2 | dm·cold2 |\n|---|---|---|---|---|---|---|")
    for lay, g in o["geometry"].items():
        L.append(row(lay, *(f"{g[k]:.2f}" for k in ("pub", "dm", "cold2", "pub·dm", "pub·cold2", "dm·cold2"))))
    L.append("\n## 3. Causal effect on next-token predictions (teacher-forced base traces)\n")
    L.append(
        "Mean change in log-probability versus base. Starters: Wait, But, Alternatively, Hmm, However, Let, Actually. "
        "Closers: </think>, So, Therefore, Thus, Final, **. Boundary = right after a sentence end or line break.\n"
    )
    heads = ["Edit", "KL on looping", "KL on productive", "Starters @ boundary", "Starters elsewhere"]
    heads += ["Closers @ boundary", "Closers elsewhere"]
    L.append(row(*heads) + "\n" + row(*["---"] * len(heads)))
    for k, c in o["causal"].items():
        deltas = (f"{c[m]:+.2f}" for m in ("starters_boundary", "starters_other", "closers_boundary", "closers_other"))
        L.append(row(k, f"{c['kl_loop']:.3f}", f"{c['kl_prod']:.3f}", *deltas))
    L.append(
        "\nProbability mass at boundaries and elsewhere (mean over positions), and closers minus starters in log-prob "
        "change at boundaries (positive = the edit shifts boundary predictions toward concluding):\n"
    )
    heads2 = [
        "Edit",
        "P(starters) @ boundary",
        "P(closers) @ boundary",
        "P(starters) elsewhere",
        "P(closers) elsewhere",
    ]
    heads2.append("Closers minus starters @ boundary")
    L.append(row(*heads2) + "\n" + row(*["---"] * len(heads2)))
    for k, c in o["causal"].items():
        masses = (
            f"{c[m]:.3f}" for m in ("starters_p_boundary", "closers_p_boundary", "starters_p_other", "closers_p_other")
        )
        L.append(row(k, *masses, f"{c['closers_boundary'] - c['starters_boundary']:+.2f}"))
    L.append("\nSingle tokens at boundaries (mean Δ log-prob):\n")
    keys = [m for m in o["causal"]["published"] if m.startswith(("S:", "C:"))]
    L.append("| Token | " + " | ".join(o["causal"]) + " |\n|---|" + "---|" * len(o["causal"]))
    for m in keys:
        L.append(
            f"| `{m[2:].replace('_boundary', '')}` | "
            + " | ".join(f"{o['causal'][k][m]:+.2f}" for k in o["causal"])
            + " |"
        )
    L.append("\n## 4. Which of the 64 edited columns write u differently on loops (top 5 per layer)\n")
    for lay, ns in o["neurons"].items():
        share = ns[-1]["share_of_gap_top5"]
        L.append(
            f"- Layer {lay} (top 5 carry {share:.0%} of the looping-minus-productive gap): "
            + ", ".join(f"col {n['column']} ({n['loop_minus_productive']:+.3f})" for n in ns[:-1])
        )
    L.append("\n## 5. Top contexts by projection onto the published direction (base MLP output)\n")
    for lay, cs in o["contexts"].items():
        L.append(f"Layer {lay}:\n")
        for c in cs[:10]:
            text = c["text"].replace("\n", "⏎").replace("|", "\\|")
            L.append(f"- {c['proj']:+.2f} {'(loop)' if c['loop'] else ''} …{text}…")
        L.append("")
    L.append("## 6. Logit lens (weak evidence at mid layers)\n")
    for k, by in o["logit_lens"].items():
        for lay, v in by.items():
            plus, minus = " ".join(repr(x) for x in v["+"][:10]), " ".join(repr(x) for x in v["-"][:10])
            L.append(f"- {k} L{lay} +u: {plus} | −u: {minus}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
