"""The reflection LM's session: read weights, run probes, write a patch. Adapted from the blog's harness prompt."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import dspy

from .hf_lm import HFLocalLM
from .probes import Probes
from .sandbox import PatchSandbox, run_fresh

BLOCK = re.compile(r"```(python|forward|patch)\s*\n(.*?)```", re.S)
# Some models emit tool-call XML instead of a fenced block; accept it rather than waste a turn.
INVOKE = re.compile(
    r'<invoke name="(python|forward|patch)">\s*<parameter name="\w+">(.*?)</parameter>\s*</invoke>', re.S
)


def find_blocks(reply: str) -> list[tuple[str, str]]:
    blocks = BLOCK.findall(reply)
    if blocks:
        return [(k, b) for k, b in blocks if b.strip() or k == "patch"] or blocks
    return [(k, b.strip("\n")) for k, b in INVOKE.findall(reply)]


logger = logging.getLogger(__name__)

SYSTEM = """\
TASK. You are given the weights of a small language model, {model_name} ({arch}). Your job is to make it better at \
grade-school math by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when it fails and \
repair that with targeted changes. No gradient descent, no training: analysis and direct edits only.

HOW IT WILL BE MEASURED. The edited model is evaluated on held-out problems with the chat template, thinking \
enabled, the user message "<problem>\\nPlease reason step by step, and put your final answer within \\boxed{{}}.", \
temperature {temperature}, top_p {top_p}, top_k {top_k}, at most {max_tokens} new tokens. Score per problem = \
0.4 * (last \\boxed{{}} answer correct) + 0.3 * (finished before the token limit) + 0.3 * (not looping, i.e. the \
distinct 8-gram ratio over the tail of the generation is >= 0.6). The dominant failure at this budget is repeating \
itself in its thinking ("Wait, let me re-read..." over and over) until it is cut off. Only weight values may change \
(same tensor names \
and shapes). Out of scope and refused by save_edit: input embedding / output head, the final norm, and multiplying \
a whole tensor by a constant. Making the model skip or truncate thinking regardless of the problem is not wanted; \
repairing a specific failure inside the thinking (such as repetition) is.

WHAT YOU HAVE.
1. A persistent Python kernel. Defined: NAMES (all tensor names), load(name) (float32 tensor: your current edit if \
any, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited(), tok (tokenizer), \
torch, np, save_artifact(name, tensor) / load_artifact(name) (persist vectors you find for use in the final patch), \
WORK (your writable folder). Gradients are disabled. Reply with exactly one block starting with ```python to run \
code; stdout is returned, cut at 10,000 characters. Print compact summaries, never whole arrays or vocabulary \
tokens. State the hypothesis of each code block in its first comment line.
2. {catalogue}
3. A forward-pass tool with a budget of {budget} passes. Reply with exactly one block starting with ```forward \
containing JSON:
 (a) {{"type": "generate", "model": "original" | "edited" | "both", "problems": ["p000", ...], "prompts": \
["your own user message", ...] (optional), "n": 1, "max_tokens": <= {max_tokens}, "activations": {{"layers": \
[0..{n_layers}]}} (optional)}}. Items x n <= {max_items}. "both" runs both models on the same items with the same \
seeds and returns the paired difference. With activations, the mean residual stream over the generated tokens \
after each requested layer (0 = embeddings) is saved as .npy, one row per item, for you to load in the kernel.
 (b) {{"type": "teacher_forced", "model": "original" | "edited", "chats": [[{{"role": "user", "content": "..."}}, \
{{"role": "assistant", "content": "..."}}], ...], "return": {{"hidden_states": [layers]}}}}: the model reads up \
to {max_items} texts you supply (for example a repetitive vs a productive continuation) and the mean residual over \
the assistant tokens is saved per layer.

HOW TO WORK. Diagnose first (read failing texts: are they cut off while repeating?). Localize causally: find the \
direction or neurons that drive the failure (for example difference of mean residuals between repetitive and \
productive continuations, projected onto MLP down_proj columns), then show that editing them changes the behaviour \
you predicted, with a dose-response. Measure paired ("both") on problems you did not use to choose the edit. Edits \
that change a tensor by a fraction of a percent usually do nothing; test strengths large enough to matter and back \
off if they hurt accuracy. Keep notes compact.

FINISH. When the budget is spent, or earlier if you are confident, reply with exactly one block starting with \
```patch containing a STANDALONE Python script that reproduces your final edit from the original weights. It may \
use load_original, save_edit, load_artifact, torch, np and literals only (no probes, no kernel variables). It is \
run in a fresh kernel and becomes the candidate that is evaluated. If the evidence supports no edit, send an EMPTY \
```patch block (that means: keep the original weights). After the patch block, one paragraph: what changed, the \
evidence, the expected effect.
"""


def _arch(lm: HFLocalLM) -> str:
    hf = getattr(lm, "hf", None)
    if hf is None:
        return f"{len(lm.names)} tensors"
    cfg = getattr(hf.config, "text_config", hf.config)
    n = getattr(cfg, "num_hidden_layers", "?")
    h = getattr(cfg, "hidden_size", "?")
    i = getattr(cfg, "intermediate_size", "?")
    lt = getattr(cfg, "layer_types", None)
    full = [k for k, t in enumerate(lt)] if lt is None else [k for k, t in enumerate(lt) if t == "full_attention"]
    kind = f"full-attention layers at {full}, the rest linear-attention" if lt else "all full attention"
    return f"{n} layers, width {h}, MLP width {i}, {kind}; tensor names like {lm.names[len(lm.names) // 2]}"


class WeightPatchProposer:
    def __init__(
        self,
        reflection_lm: dspy.LM,
        lm: HFLocalLM,
        probes: Probes,
        work_dir: str,
        max_turns: int = 40,
        max_code_runs: int = 30,
        lm_timeout: float = 600.0,
        lm_attempts: int = 3,
    ):
        self.reflection_lm, self.lm, self.probes, self.work_dir = reflection_lm, lm, probes, work_dir
        self.max_turns, self.max_code_runs = max_turns, max_code_runs
        self.lm_timeout, self.lm_attempts = lm_timeout, lm_attempts
        self.round = 0
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _ask(self, messages: list[dict[str, str]]) -> str:
        """Call the reflection LM with a wall-clock timeout; a hung HTTP request must not stall the run."""
        last: Exception | None = None
        for attempt in range(1, self.lm_attempts + 1):
            fut = self._pool.submit(self.reflection_lm, messages=messages)
            try:
                return fut.result(timeout=self.lm_timeout)[0]
            except FutureTimeout as e:
                last = e
                logger.warning(
                    "reflection LM timed out after %.0fs (attempt %d/%d)", self.lm_timeout, attempt, self.lm_attempts
                )
            except Exception as e:  # noqa: BLE001
                last = e
                logger.warning("reflection LM call failed (attempt %d/%d): %s", attempt, self.lm_attempts, e)
                time.sleep(5 * attempt)
        raise RuntimeError(f"reflection LM failed after {self.lm_attempts} attempts") from last

    def _system(self) -> str:
        return SYSTEM.format(
            model_name=self.lm.model_name,
            arch=_arch(self.lm),
            temperature=self.lm.kwargs["temperature"],
            top_p=self.lm.top_p,
            top_k=self.lm.top_k,
            max_tokens=self.probes.program.max_new_tokens or self.lm.kwargs["max_tokens"],
            catalogue=self.probes.catalogue(),
            budget=self.probes.budget,
            n_layers=len(self.lm._layers()),
            max_items=self.probes.max_items,
        )

    def _opening(self, candidate: dict[str, str], reflective: list[dict]) -> str:
        cur = candidate.get("weight_patch", "").strip()
        parts = []
        if cur:
            parts.append(
                "CURRENT BEST PATCH (already applied in your kernel as the current edit; improve or replace it):\n"
                "```python\n" + cur + "\n```"
            )
        else:
            parts.append("CURRENT BEST PATCH: none (original weights).")
        parts.append(f"RECENT EVALUATION of the current patch on {len(reflective)} problems:")
        for r in reflective[:8]:
            prob = str(r["Inputs"].get("problem", ""))[:160]
            tail = str(r["Generated Outputs"])[-240:]
            parts.append(f"- problem: {prob!r}\n  output tail: {tail!r}\n  feedback: {r['Feedback']}")
        parts.append(f"Forward passes available this session: {self.probes.left}. Begin.")
        return "\n".join(parts)

    def __call__(self, candidate: dict[str, str], reflective_dataset, components_to_update) -> dict[str, str]:
        self.round += 1
        self.probes.used = 0
        rdir = os.path.join(self.work_dir, f"round_{self.round:02d}")
        os.makedirs(rdir, exist_ok=True)
        self.probes.work_dir = rdir
        sb = PatchSandbox(self.lm, self.work_dir)
        self.probes.sandbox = sb
        if candidate.get("weight_patch", "").strip():
            sb.run(candidate["weight_patch"])

        messages = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": self._opening(candidate, reflective_dataset.get("weight_patch", []))},
        ]
        transcript = open(os.path.join(rdir, "transcript.md"), "w")
        code_runs = 0
        final: str | None = None
        for turn in range(self.max_turns):
            reply = self._ask(messages)
            messages.append({"role": "assistant", "content": reply})
            transcript.write(f"\n\n## assistant (turn {turn + 1})\n{reply}")
            transcript.flush()
            blocks = find_blocks(reply)
            if not blocks:
                if self.probes.left <= 0 and sb.edits:
                    final = sb.export_script()
                    break
                obs = "No code block found. Reply with exactly one ```python, ```forward or ```patch block."
            else:
                # Run every block in order: models often pair an edit (python) with a probe (forward) in one reply.
                obs_parts = []
                for kind, body in blocks:
                    if kind == "patch":
                        if not body.strip():
                            final = ""
                            break
                        fresh, res = run_fresh(self.lm, body, self.work_dir)
                        if res.ok and fresh.edits:
                            final = body
                            break
                        why = res.error or "no save_edit calls"
                        obs_parts.append(
                            f"patch failed when run in a fresh kernel:\n{why}\nFix and resend a ```patch block."
                        )
                    elif kind == "python":
                        code_runs += 1
                        if code_runs > self.max_code_runs:
                            obs_parts.append(
                                f"Code run limit ({self.max_code_runs}) reached. Send your ```patch block now."
                            )
                        else:
                            res = sb.run(body)
                            err = f"\nERROR:\n{res.error}" if res.error else ""
                            obs_parts.append((res.stdout or "(no output)") + err + f"\n[edited: {res.edits}]")
                    else:
                        try:
                            obs_parts.append(self.probes.dispatch(json.loads(body)))
                        except json.JSONDecodeError as e:
                            obs_parts.append(f"error: forward block is not valid JSON ({e})")
                if final is not None:
                    break
                obs = "\n\n".join(obs_parts)
            if self.probes.left <= 0 and final is None:
                obs += "\n\nBudget spent. Send your final ```patch block (standalone script) now."
            messages.append({"role": "user", "content": obs})
            transcript.write(f"\n\n## tool\n{obs}")
            transcript.flush()
        if final is None:
            final = sb.export_script() if sb.edits else candidate.get("weight_patch", "")
        transcript.write(f"\n\n## final patch\n```python\n{final}\n```\n")
        transcript.close()
        self.lm.reset()
        with open(os.path.join(rdir, "patch.py"), "w") as f:
            f.write(final)
        json.dump(
            {
                "round": self.round,
                "turns": len(messages) // 2,
                "code_runs": code_runs,
                "passes": self.probes.used,
                "time": time.time(),
            },
            open(os.path.join(rdir, "meta.json"), "w"),
        )
        return {"weight_patch": final}
