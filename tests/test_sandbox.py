import pytest
import torch

from dspy_weightspace.sandbox import EditRejected, PatchSandbox, run_fresh

from .conftest import FakeLM

L1 = "model.language_model.layers.1.mlp.down_proj.weight"


def test_noop_patch_changes_nothing(tmp_path):
    lm = FakeLM()
    sb, res = run_fresh(lm, "# nothing\n", str(tmp_path))
    assert res.ok and sb.edits == {}
    sb.apply()
    assert lm.changed_entries() == 0


def test_save_edit_counts_changed_entries(tmp_path):
    lm = FakeLM()
    sb = PatchSandbox(lm, str(tmp_path))
    res = sb.run(f"w = load_original({L1!r}); w[:, :4] = 0; save_edit({L1!r}, w)")
    assert res.ok, res.error
    sb.apply()
    assert lm.edited() == [L1]
    assert lm.changed_entries() == 8 * 4
    lm.reset()
    assert lm.changed_entries() == 0


def test_forbidden_tensors_rejected(tmp_path):
    lm = FakeLM()
    sb = PatchSandbox(lm, str(tmp_path))
    for name in ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight"):
        with pytest.raises(EditRejected):
            sb.save_edit(name, lm.original(name) * 2)


def test_scalar_multiple_rejected(tmp_path):
    lm = FakeLM()
    sb = PatchSandbox(lm, str(tmp_path))
    with pytest.raises(EditRejected):
        sb.save_edit(L1, lm.original(L1) * 1.3)


def test_shape_and_nonfinite_rejected(tmp_path):
    lm = FakeLM()
    sb = PatchSandbox(lm, str(tmp_path))
    with pytest.raises(EditRejected):
        sb.save_edit(L1, torch.zeros(3, 3))
    bad = lm.original(L1)
    bad[0, 0] = float("nan")
    with pytest.raises(EditRejected):
        sb.save_edit(L1, bad)


def test_export_script_roundtrip(tmp_path):
    lm = FakeLM()
    sb = PatchSandbox(lm, str(tmp_path))
    sb.run(f"w = load_original({L1!r}); w[0] = 0; save_edit({L1!r}, w)")
    script = sb.export_script()
    sb2, res = run_fresh(lm, script, str(tmp_path))
    assert res.ok, res.error
    assert torch.equal(sb2.edits[L1], sb.edits[L1])


def test_stdout_truncated(tmp_path):
    sb = PatchSandbox(FakeLM(), str(tmp_path))
    res = sb.run("print('x' * 20000)")
    assert len(res.stdout) < 10_200 and "cut at" in res.stdout


def test_time_limit_interrupts_python_loop(tmp_path):
    sb = PatchSandbox(FakeLM(), str(tmp_path), time_limit_s=1)
    res = sb.run("i = 0\nwhile True:\n    i += 1\n")
    assert not res.ok and "exceeded" in res.error
    # kernel still usable afterwards
    assert sb.run("print(i > 0)").stdout.strip() == "True"
