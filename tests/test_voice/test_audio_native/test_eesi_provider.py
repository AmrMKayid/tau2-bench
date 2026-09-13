"""Unit tests for the EESI realtime provider's connect handshake and knobs.

These run against a fake websocket, so they need no EESI credentials and no
network. The live handshake is covered by ``test_provider_suite.py``.
"""

import asyncio
import json
from collections import deque
from typing import Any, Dict, List

import pytest
from websockets.protocol import State

from tau2.voice.audio_native.eesi import provider as eesi_provider
from tau2.voice.audio_native.eesi.provider import EesiRealtimeProvider
from tau2.voice.audio_native.openai.provider import OpenAIVADConfig

GATEWAY_UUID = "2b6b1c1e-6d1f-4c0e-9c3a-8a0a2d6f4e11"
BASE_URL = "wss://eesi.test/v1/realtime"

MARKER = {"type": "eesi.synthetic_speech_marker", "standard": "EU AI Act 50(2)"}
DISCLOSURE = {"type": "eesi.recording_disclosure", "recorded": True}
GATEWAY_SESSION = {
    "type": "eesi.session",
    "session_id": GATEWAY_UUID,
    "resume_token": "rt_abc",
}
SESSION_CREATED_NO_ID = {"type": "session.created", "session": {"id": None}}


class FakeWebSocket:
    """Scripted server: hands out queued frames, records what the client sent."""

    def __init__(self, frames: List[Dict[str, Any]]):
        self.state = State.OPEN
        self.incoming = deque(json.dumps(f) for f in frames)
        self.sent: List[Dict[str, Any]] = []

    def queue(self, frame: Dict[str, Any]) -> None:
        self.incoming.append(json.dumps(frame))

    async def recv(self) -> str:
        if not self.incoming:
            raise AssertionError("client waited for a frame the script never sent")
        return self.incoming.popleft()

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def close(self) -> None:
        self.state = State.CLOSED


@pytest.fixture
def fake_connect(monkeypatch):
    """Patch ``websockets.connect`` as the provider sees it; returns a recorder."""
    calls: Dict[str, Any] = {}

    def install(frames: List[Dict[str, Any]]) -> FakeWebSocket:
        ws = FakeWebSocket(frames)

        async def _connect(url, **kwargs):
            calls["url"] = url
            calls["headers"] = kwargs.get("additional_headers")
            return ws

        monkeypatch.setattr(eesi_provider.websockets, "connect", _connect)
        return ws

    calls["install"] = install
    return calls


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests decide these themselves; nothing from .env may leak in."""
    monkeypatch.delenv("EESI_REALTIME_SOURCE", raising=False)
    monkeypatch.delenv("EESI_NUR_CONFIGURE_JSON", raising=False)


def make_provider(base_url: str = BASE_URL) -> EesiRealtimeProvider:
    return EesiRealtimeProvider(
        api_key="sk-eesi-test", model="nur-realtime-v1", base_url=base_url
    )


class TestSessionId:
    def test_gateway_session_frame_sets_session_id(self, fake_connect):
        fake_connect["install"](
            [MARKER, DISCLOSURE, GATEWAY_SESSION, SESSION_CREATED_NO_ID]
        )
        provider = make_provider()

        asyncio.run(provider.connect())

        assert provider.session_id == GATEWAY_UUID
        assert provider.gateway_session_id == GATEWAY_UUID

    def test_session_created_does_not_overwrite_gateway_id(self, fake_connect):
        created_with_id = {"type": "session.created", "session": {"id": "sess_x"}}
        fake_connect["install"]([GATEWAY_SESSION, created_with_id])
        provider = make_provider()

        asyncio.run(provider.connect())

        assert provider.session_id == GATEWAY_UUID

    def test_session_created_id_is_the_fallback_without_a_gateway(self, fake_connect):
        # A bare speech orchestrator (s2s pool) sends no eesi.session frame.
        created_with_id = {"type": "session.created", "session": {"id": "sess_x"}}
        fake_connect["install"]([created_with_id])
        provider = make_provider()

        asyncio.run(provider.connect())

        assert provider.session_id == "sess_x"
        assert provider.gateway_session_id is None

    def test_reconnect_does_not_keep_a_stale_id(self, fake_connect):
        fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider()
        asyncio.run(provider.connect())
        asyncio.run(provider.disconnect())

        fresh = {
            **GATEWAY_SESSION,
            "session_id": "0f0f0f0f-0000-4000-8000-000000000002",
        }
        fake_connect["install"]([fresh, SESSION_CREATED_NO_ID])
        asyncio.run(provider.connect())

        assert provider.session_id == fresh["session_id"]

    def test_error_frame_refuses_the_session(self, fake_connect):
        fake_connect["install"]([{"type": "error", "error": {"message": "nope"}}])
        provider = make_provider()

        with pytest.raises(RuntimeError, match="nope"):
            asyncio.run(provider.connect())


class TestUrl:
    def test_source_defaults_to_tau2(self, fake_connect):
        fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider()

        asyncio.run(provider.connect())

        assert fake_connect["url"] == f"{BASE_URL}?model=nur-realtime-v1&source=tau2"
        assert fake_connect["headers"] == {"Authorization": "Bearer sk-eesi-test"}

    def test_source_honours_env(self, fake_connect, monkeypatch):
        monkeypatch.setenv("EESI_REALTIME_SOURCE", "tau2 loop/a")
        fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider()

        asyncio.run(provider.connect())

        assert fake_connect["url"].endswith("&source=tau2+loop%2Fa")
        assert "model=nur-realtime-v1" in fake_connect["url"]

    def test_base_url_with_query_joins_with_ampersand(self, fake_connect):
        fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider(base_url=f"{BASE_URL}?token=abc")

        asyncio.run(provider.connect())

        assert (
            fake_connect["url"]
            == f"{BASE_URL}?token=abc&model=nur-realtime-v1&source=tau2"
        )


def _configure(provider: EesiRealtimeProvider) -> None:
    asyncio.run(
        provider.configure_session(
            system_prompt="Be brief.",
            tools=[],
            vad_config=OpenAIVADConfig(),
            modality="audio",
        )
    )


class TestNurConfigure:
    def test_sends_configure_frame_after_session_update(
        self, fake_connect, monkeypatch
    ):
        monkeypatch.setenv(
            "EESI_NUR_CONFIGURE_JSON",
            '{"pacing": "quick", "backchannels": false, "language": "en"}',
        )
        ws = fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider()
        asyncio.run(provider.connect())
        ws.queue({"type": "session.updated", "session": {}})

        _configure(provider)

        assert [f["type"] for f in ws.sent] == ["session.update", "eesi.nur.configure"]
        assert ws.sent[1] == {
            "type": "eesi.nur.configure",
            "pacing": "quick",
            "backchannels": False,
            "language": "en",
        }
        assert ws.sent[0]["session"]["instructions"] == "Be brief."
        # Fire-and-forget: nothing left waiting on an ack that never comes.
        assert not ws.incoming

    def test_sends_nothing_when_env_unset(self, fake_connect):
        ws = fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider()
        asyncio.run(provider.connect())
        ws.queue({"type": "session.updated", "session": {}})

        _configure(provider)

        assert [f["type"] for f in ws.sent] == ["session.update"]

    @pytest.mark.parametrize("raw", ["not json", "[1, 2]", "   "])
    def test_unusable_env_sends_nothing(self, fake_connect, monkeypatch, raw):
        monkeypatch.setenv("EESI_NUR_CONFIGURE_JSON", raw)
        ws = fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider()
        asyncio.run(provider.connect())
        ws.queue({"type": "session.updated", "session": {}})

        _configure(provider)

        assert [f["type"] for f in ws.sent] == ["session.update"]

    def test_browser_only_keys_are_dropped(self, fake_connect, monkeypatch):
        monkeypatch.setenv(
            "EESI_NUR_CONFIGURE_JSON",
            '{"pacing": "patient", "turn_control": true, "server_tasks": true}',
        )
        ws = fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider()
        asyncio.run(provider.connect())
        ws.queue({"type": "session.updated", "session": {}})

        _configure(provider)

        assert ws.sent[1] == {"type": "eesi.nur.configure", "pacing": "patient"}

    def test_type_cannot_be_overridden_by_env(self, fake_connect, monkeypatch):
        monkeypatch.setenv(
            "EESI_NUR_CONFIGURE_JSON", '{"type": "session.update", "pacing": "quick"}'
        )
        ws = fake_connect["install"]([GATEWAY_SESSION, SESSION_CREATED_NO_ID])
        provider = make_provider()
        asyncio.run(provider.connect())
        ws.queue({"type": "session.updated", "session": {}})

        _configure(provider)

        assert ws.sent[1]["type"] == "eesi.nur.configure"
