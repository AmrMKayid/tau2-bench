"""``TAU2_EESI_INSTRUCTIONS_PREFIX_FILE`` prepends a file to the EESI agent prompt.

Lives beside the EESI provider tests rather than in
``test_streaming/test_discrete_time_audio_native_agent.py`` because that module
no longer collects (it imports a ``tick_runner`` that upstream removed).
"""

from unittest.mock import MagicMock

import pytest

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent

POLICY = "You are the airline desk. Never promise a refund you cannot see."
PREFIX = "Answer in one short sentence."


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("TAU2_EESI_INSTRUCTIONS_PREFIX_FILE", raising=False)


def make_agent(provider: str) -> DiscreteTimeAudioNativeAgent:
    adapter = MagicMock()
    adapter.is_connected = False
    return DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy=POLICY,
        tick_duration_ms=1000,
        modality="audio",
        adapter=adapter,
        provider=provider,
    )


def test_prefix_applied_for_eesi(tmp_path, monkeypatch):
    baseline = make_agent("eesi").system_prompt
    prefix_file = tmp_path / "prefix.txt"
    prefix_file.write_text(f"{PREFIX}\n")
    monkeypatch.setenv("TAU2_EESI_INSTRUCTIONS_PREFIX_FILE", str(prefix_file))

    prompt = make_agent("eesi").system_prompt

    assert prompt == f"{PREFIX}\n\n{baseline}"
    assert POLICY in prompt


def test_prefix_ignored_for_other_providers(tmp_path, monkeypatch):
    baseline = make_agent("openai").system_prompt
    prefix_file = tmp_path / "prefix.txt"
    prefix_file.write_text(PREFIX)
    monkeypatch.setenv("TAU2_EESI_INSTRUCTIONS_PREFIX_FILE", str(prefix_file))

    assert make_agent("openai").system_prompt == baseline


def test_missing_file_leaves_prompt_alone(tmp_path, monkeypatch):
    baseline = make_agent("eesi").system_prompt
    monkeypatch.setenv(
        "TAU2_EESI_INSTRUCTIONS_PREFIX_FILE", str(tmp_path / "missing.txt")
    )

    assert make_agent("eesi").system_prompt == baseline


def test_empty_file_leaves_prompt_alone(tmp_path, monkeypatch):
    baseline = make_agent("eesi").system_prompt
    prefix_file = tmp_path / "prefix.txt"
    prefix_file.write_text("  \n")
    monkeypatch.setenv("TAU2_EESI_INSTRUCTIONS_PREFIX_FILE", str(prefix_file))

    assert make_agent("eesi").system_prompt == baseline
