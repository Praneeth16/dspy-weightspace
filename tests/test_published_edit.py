import numpy as np
import torch

from dspy_weightspace.published_edit import bf16_bits, bits_to_bf16, sha256


def test_bf16_bit_roundtrip():
    t = torch.randn(64, 32).to(torch.bfloat16)
    bits = bf16_bits(t)
    assert bits.dtype == np.uint16 and bits.shape == (64, 32)
    assert torch.equal(bits_to_bf16(bits), t)


def test_sparse_override_changes_only_listed_entries():
    t = torch.randn(64, 32).to(torch.bfloat16)
    base = bf16_bits(t)
    flat = base.copy().reshape(-1)
    idx = np.array([0, 5, 100, 2047])
    flat[idx] = np.uint16(0x3F80)  # 1.0 in bf16
    edited = flat.reshape(base.shape)
    assert int((edited != base).sum()) <= 4
    assert bits_to_bf16(edited).float().reshape(-1)[idx].tolist() == [1.0] * 4
    assert sha256(edited) != sha256(base)
