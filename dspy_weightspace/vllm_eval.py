"""Blog-protocol evaluation with vLLM: thinking on, T=0.6, top_p 0.95, top_k 20, presence_penalty 1.5, 32,768 tokens.

Matches the protocol in RikiyaT/lm_understand-logs exp8_math/code/evaluate.py and the ML-Intern-lab / akhaliq
verification (vLLM, seed 0, last \\boxed{} graded). Use on a GPU box to compare a patched checkpoint with the base
and the published edit at full budget. Needs: pip install vllm math-verify

    python -m dspy_weightspace.vllm_eval --model Qwen/Qwen3.5-0.8B --n 48 --samples 2
    python -m dspy_weightspace.vllm_eval --model /path/to/patched_ckpt --n 48 --samples 2
"""

from __future__ import annotations

import argparse
import json
import statistics

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="HuggingFaceH4/MATH-500", help="or openai/gsm8k")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--out", default="eval_vllm.jsonl")
    a = ap.parse_args()

    from datasets import load_dataset
    from math_verify import parse, verify
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if a.dataset == "openai/gsm8k":
        ds = load_dataset(a.dataset, "main", split="test").select(range(a.n))
        probs = [(r["question"], r["answer"].rsplit("####", 1)[-1].strip()) for r in ds]
    else:
        ds = load_dataset(a.dataset, split="test").select(range(a.n))
        probs = [(r["problem"], r["answer"]) for r in ds]

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": f"{p}\n{INSTRUCTION}"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p, _ in probs
    ]
    # disable_log_stats=False: periodic throughput lines, since tqdm's carriage returns never reach batch-job logs.
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_tokens + 2048, seed=0, disable_log_stats=False)
    sp = SamplingParams(
        n=a.samples, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=a.max_tokens, seed=0
    )
    outs = llm.generate(prompts, sp)

    correct, cut, rows = [], [], []
    for (p, gold), o in zip(probs, outs, strict=True):
        for c in o.outputs:
            text = c.text
            # As in the post's evaluate.py: thinking that never closed gives no answer, even if it wrote a \boxed{}.
            final = text.rsplit("</think>", 1)[-1] if "</think>" in text else ""
            ok = bool(verify(parse(f"${gold}$"), parse(final)))
            correct.append(ok)
            cut.append(c.finish_reason == "length")
            rows.append(
                {"problem": p[:200], "gold": gold, "correct": ok, "finish": c.finish_reason, "tokens": len(c.token_ids)}
            )
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "model": a.model,
        "generations": len(rows),
        "accuracy": statistics.mean(correct),
        "cut_at_limit": statistics.mean(cut),
        "mean_tokens": statistics.mean(r["tokens"] for r in rows),
    }
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
