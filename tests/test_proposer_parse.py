from dspy_weightspace.proposer import find_blocks


def test_fenced_block():
    assert find_blocks("hi\n```python\nprint(1)\n```\n") == [("python", "print(1)\n")]


def test_invoke_xml_fallback():
    reply = '<invoke name="python">\n<parameter name="code">x = 1\nprint(x)</parameter>\n</invoke>\n```\n'
    assert find_blocks(reply) == [("python", "x = 1\nprint(x)")]


def test_empty_fence_after_invoke_is_ignored():
    reply = '<invoke name="forward">\n<parameter name="json">{"type": "generate"}</parameter>\n</invoke>\n```\n\n```'
    assert find_blocks(reply)[0][0] == "forward"


def test_empty_patch_block_is_kept():
    assert find_blocks("```patch\n```") == [("patch", "")]
