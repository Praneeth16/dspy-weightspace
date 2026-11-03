"""Reflection LM constructors. Any dspy.LM works; these two are the ones the notebooks use."""

from __future__ import annotations

import os

import dspy


def anthropic_lm(model: str = "anthropic/claude-fable-5-1", **kw) -> dspy.LM:
    return dspy.LM(model, max_tokens=kw.pop("max_tokens", 8000), cache=False, **kw)


def databricks_lm(endpoint: str, profile: str | None = None, **kw) -> dspy.LM:
    """Foundation Model API endpoint through the OpenAI-compatible route, authenticated by the workspace profile
    (or the notebook's own context when running on Databricks). Tokens are short-lived: for runs over an hour,
    call this again."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    token = w.config.authenticate()["Authorization"].split(" ", 1)[1]
    # Claude 5.x endpoints reject the temperature parameter; leave it unset unless the caller insists.
    return dspy.LM(
        f"openai/{endpoint}",
        api_base=f"{w.config.host}/serving-endpoints",
        api_key=token,
        max_tokens=kw.pop("max_tokens", 8000),
        cache=False,
        **kw,
    )


def from_env() -> dspy.LM:
    """REFLECTION_LM=databricks:<endpoint> | anthropic:<model>."""
    spec = os.environ.get("REFLECTION_LM", "databricks:databricks-claude-sonnet-5-5")
    kind, _, name = spec.partition(":")
    if kind == "databricks":
        return databricks_lm(name, profile=os.environ.get("DATABRICKS_CONFIG_PROFILE"))
    return anthropic_lm(name)
