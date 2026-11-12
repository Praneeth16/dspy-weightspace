"""Generate with vLLM while the optimizer edits weights held by the HF model.

HF generate on this hybrid model runs ~10x slower than vLLM 0.33, too slow to select edits on long MATH traces.
HFLocalLM(gen_backend=VLLMBackend(...)) keeps the HF model for the sandbox and teacher-forced probes and sends every
generation here; apply_patch and reset copy the touched tensors into vLLM's copy of the model in place.
"""

from __future__ import annotations

import os
import shutil

import torch

from .hf_lm import Completion

# The final eval's thinking protocol (vllm_eval.PROTOCOLS["think35"] plus its presence penalty), so search and test
# sample the same way.
THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def drop_stale_flash_attn() -> None:
    """The image's flash-attn is compiled against its own torch; a newer vLLM brings a newer torch, and vLLM then
    imports the stale extension (undefined symbol) for the vision tower's rotary embedding. vLLM ships its own
    attention kernels, so the package is removed instead of rebuilt."""
    from importlib import metadata

    try:
        dist = metadata.distribution("flash-attn")
    except metadata.PackageNotFoundError:
        return
    version, root = dist.version, dist.locate_file("")
    files = [root / f for f in dist.files or []]
    for path in files:
        if path.is_file():
            path.unlink()
    for name in ("flash_attn", "hopper"):
        shutil.rmtree(root / name, ignore_errors=True)
    for d in sorted({p.parent for p in files}, key=lambda d: len(d.parts), reverse=True):
        if d.is_dir() and d != root and not any(d.iterdir()):
            d.rmdir()  # an empty .dist-info would still register as an installed distribution
    print(f"removed stale flash-attn {version}", flush=True)


def layer_suffix(name: str) -> str:
    """'model.layers.15.mlp.down_proj.weight' -> 'layers.15.mlp.down_proj.weight', unique in vLLM's naming too."""
    return name[name.index("layers.") :]


# vLLM stores some projections stacked row-wise in one matrix (packed_modules_mapping in its qwen3_5.py). A patch
# that edits one part has to land in that part's rows of the stacked matrix.
PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


def locate(params: dict, suffix: str, rows: int):
    """(parameter name, row offset) in vLLM's model for a tensor the HF model calls `suffix`."""
    hits = [k for k in params if k.endswith(suffix)]
    if len(hits) == 1:
        return hits[0], 0
    module, part, leaf = suffix.rsplit(".", 2)
    for packed, parts in PACKED.items():
        if part not in parts:
            continue
        hits = [k for k in params if k.endswith(f"{module}.{packed}.{leaf}")]
        if len(hits) != 1:
            break
        total, i = params[hits[0]].shape[0], parts.index(part)
        # The first part starts at row 0 and the last ends at the last row. The only middle part is k_proj, which
        # has as many rows as the v_proj after it.
        offset = 0 if i == 0 else total - rows * (len(parts) - i)
        if offset < 0 or offset + rows > total:
            raise ValueError(f"{suffix}: {rows} rows do not fit in {hits[0]} ({total} rows)")
        return hits[0], offset
    raise KeyError(f"{suffix}: no matching vLLM parameter")


class VLLMBackend:
    def __init__(
        self,
        model_dir: str,
        max_model_len: int = 34816,
        gpu_memory_utilization: float = 0.5,
        sampling: dict | None = None,
    ):
        # Engine in this process: no spawned child re-importing the (unguarded) runner script, and apply_model
        # reaches the model directly instead of pickling the new tensors across processes.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        drop_stale_flash_attn()
        from vllm import LLM, SamplingParams

        self._sampling_params = SamplingParams
        self.sampling = dict(sampling or THINK35)
        # No prefix caching: cached prompt KV would have been computed with the weights before the last patch.
        self.llm = LLM(
            model=model_dir,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            seed=0,
        )

    def generate(
        self, prompts: list[str], max_new_tokens: int, temperature: float | None = None, seed: int | None = None
    ) -> list[Completion]:
        kw = dict(self.sampling)
        if temperature is not None:
            kw["temperature"] = temperature
        sp = self._sampling_params(max_tokens=max_new_tokens, seed=seed, **kw)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [
            Completion(
                o.outputs[0].text,
                "length" if o.outputs[0].finish_reason == "length" else "stop",
                len(o.outputs[0].token_ids),
            )
            for o in outs
        ]

    def set_weights(self, tensors: dict[str, torch.Tensor]) -> None:
        payload = {layer_suffix(n): t.detach().to("cpu", torch.bfloat16).contiguous() for n, t in tensors.items()}

        def copy_in(model):
            params = dict(model.named_parameters())
            # Find every target first: a name that cannot be placed must fail before any tensor is copied.
            targets = []
            for suffix, t in payload.items():
                name, offset = locate(params, suffix, t.shape[0])
                target = params[name].data[offset : offset + t.shape[0]]
                if tuple(target.shape) != tuple(t.shape):
                    raise ValueError(f"{name}: vLLM shape {tuple(target.shape)} != {tuple(t.shape)}")
                targets.append((name, target, t))
            for _, target, t in targets:
                target.copy_(t.to(target.device, target.dtype))
            return [name for name, _, _ in targets]

        self.llm.apply_model(copy_in)
