"""EESI realtime provider (api.eesi.ai / nur-realtime-v1).

EESI's ``/v1/realtime`` speaks the GA OpenAI Realtime protocol: the same
``session.update`` shape, the same ``input_audio_buffer.append``, the same
``response.output_audio.delta`` / ``response.output_audio_transcript.delta`` /
``response.function_call_arguments.done`` server events, and the same
``function_call_output`` + ``response.create`` tool-result handshake. So this
provider is the OpenAI one pointed at a different host, not a reimplementation
of it — anything that drifts in the OpenAI provider should drift here too.

Three things differ:

- **The endpoint is configurable.** EESI runs a dev deployment, a production
  one and a bare speech orchestrator (the ``s2s`` pool, no gateway in front),
  and the same benchmark has to be runnable against each. ``EESI_REALTIME_URL``
  selects one; the default is dev. The URL also carries a free-form ``source``
  (``EESI_REALTIME_SOURCE``, default ``tau2``) that the gateway records on the
  session row, so EESI-side dashboards can tell benchmark traffic apart.
- **The greeting is not the first frame.** EESI's gateway sends its own
  declarations before it relays anything from the speech server: the EU AI Act
  Art. 50(2) synthetic-speech marker, the recording disclosure, and the
  ``eesi.session`` handle used for resume. ``session.created`` arrives *after*
  them, so connect scans for it instead of asserting on frame one. The
  ``eesi.session`` frame is also where the session's identity lives: its
  ``session_id`` is the ``speech_sessions`` row uuid on the EESI API
  (``GET /v1/speech/sessions/{id}``), which is what lets a benchmark run be
  joined to EESI's recording, judge and runtime bundle. The speech server's
  ``session.created`` carries no id of its own.
- **Nur has knobs the OpenAI session shape cannot express.** After
  ``session.updated`` the gateway accepts one ``eesi.nur.configure`` frame
  (pacing, backchannels, language); ``EESI_NUR_CONFIGURE_JSON`` sends it.
"""

import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import websockets
from loguru import logger

from tau2.config import (
    DEFAULT_EESI_REALTIME_BASE_URL,
    DEFAULT_EESI_REALTIME_MODEL,
)
from tau2.data_model.audio import AudioFormat
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
)

#: Gateway frames that may precede ``session.created``. Scanning is bounded so a
#: misconfigured endpoint fails at connect rather than hanging the simulation.
MAX_GREETING_FRAMES = 16

#: ``source`` recorded on the EESI session row when ``EESI_REALTIME_SOURCE`` is
#: unset. The DUET seat uses ``duet-run``; the benchmark runner is ``tau2``.
DEFAULT_REALTIME_SOURCE = "tau2"

#: ``eesi.nur.configure`` keys that only make sense for the browser client:
#: ``turn_control`` hands turn-taking to a UI that a headless socket does not
#: have, and ``server_tasks`` schedules work the runner never reads back. Either
#: one set to true leaves the benchmark socket waiting on frames that never
#: come, so they are dropped rather than forwarded.
_BROWSER_ONLY_NUR_KEYS = ("turn_control", "server_tasks")


class EesiRealtimeProvider(OpenAIRealtimeProvider):
    """OpenAI Realtime provider pointed at an EESI realtime endpoint."""

    BASE_URL = DEFAULT_EESI_REALTIME_BASE_URL
    DEFAULT_MODEL = DEFAULT_EESI_REALTIME_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the EESI realtime provider.

        Args:
            api_key: EESI API key (``sk-eesi-…``). Defaults to ``EESI_API_KEY``.
            model: Model id. Defaults to ``EESI_REALTIME_MODEL`` then
                ``nur-realtime-v1``.
            reasoning_effort: Passed through as ``session.reasoning.effort``.
            base_url: WebSocket endpoint. Defaults to ``EESI_REALTIME_URL``
                then the dev deployment.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("EESI_API_KEY")
        if not self.api_key:
            raise ValueError("EESI API key not provided. Set EESI_API_KEY env var.")

        self.base_url = base_url or os.environ.get("EESI_REALTIME_URL") or self.BASE_URL
        self.model = (
            model or os.environ.get("EESI_REALTIME_MODEL") or self.DEFAULT_MODEL
        )
        self.reasoning_effort = reasoning_effort
        self.ws = None
        self._current_vad_config = None
        from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT

        self._audio_format = TELEPHONY_AUDIO_FORMAT
        #: Session id recorded on the SimulationRun (``provider_session_id``).
        #: For EESI this is the gateway's row uuid, so it joins to
        #: ``GET /v1/speech/sessions/{id}``.
        self.session_id: Optional[str] = None
        #: The ``eesi.session`` handle verbatim; None against a bare speech
        #: orchestrator, which has no gateway in front to send one.
        self.gateway_session_id: Optional[str] = None

    def _build_url(self) -> str:
        """Endpoint plus ``model`` and ``source`` query parameters.

        ``base_url`` may already carry a query (a pinned pool, a tunnel with a
        token), so the parameters are joined with ``&`` in that case.
        """
        source = os.environ.get("EESI_REALTIME_SOURCE") or DEFAULT_REALTIME_SOURCE
        query = urlencode({"model": self.model, "source": source})
        separator = "&" if "?" in self.base_url else "?"
        return f"{self.base_url}{separator}{query}"

    @websocket_retry
    async def connect(self) -> None:
        """Open the socket and wait for ``session.created``.

        Unlike OpenAI's endpoint, EESI's gateway speaks first: it emits the
        synthetic-speech marker, the recording disclosure and the
        ``eesi.session`` resume handle before relaying the speech server's
        ``session.created``. The handle's ``session_id`` becomes
        ``self.session_id``; the other frames are logged and skipped.

        Raises:
            RuntimeError: If no ``session.created`` arrives within
                ``MAX_GREETING_FRAMES``, or the endpoint reports an error.
        """
        if self.is_connected:
            return

        # A reconnect is a new gateway row; never let a stale id survive into
        # it, or the SimulationRun would join to the wrong session.
        self.session_id = None
        self.gateway_session_id = None

        url = self._build_url()
        headers = {"Authorization": f"Bearer {self.api_key}"}

        self.ws = await websockets.connect(url, additional_headers=headers)

        for _ in range(MAX_GREETING_FRAMES):
            data = json.loads(await self.ws.recv())
            event_type = data.get("type")

            if event_type == "eesi.session":
                self.gateway_session_id = data.get("session_id")
                if self.gateway_session_id:
                    self.session_id = self.gateway_session_id
                logger.info(
                    f"EESI Realtime API: gateway session "
                    f"session_id={self.gateway_session_id} "
                    f"(resume_token={'set' if data.get('resume_token') else 'none'})"
                )
                continue

            if event_type == "session.created":
                # EESI's speech server leaves session.id empty; only fall back
                # to it when no gateway handle arrived (bare s2s pool).
                if self.session_id is None:
                    self.session_id = data.get("session", {}).get("id")
                logger.info(
                    f"EESI Realtime API: session created "
                    f"(url={self.base_url}, model={self.model}, "
                    f"session_id={self.session_id})"
                )
                return

            if event_type == "error":
                message = data.get("error", {}).get("message", "unknown error")
                raise RuntimeError(f"EESI realtime refused the session: {message}")

            # The marker and the recording disclosure carry compliance state.
            # Neither changes how the benchmark drives the socket, so note
            # and move on.
            logger.debug(f"EESI Realtime API: greeting frame {event_type}")

        raise RuntimeError(
            f"No session.created within {MAX_GREETING_FRAMES} frames from {self.base_url}"
        )

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: OpenAIVADConfig,
        modality: str = "text",
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        """OpenAI's ``session.update`` handshake, then Nur's own knobs.

        The gateway only honours ``eesi.nur.configure`` once the session is
        configured, so it is sent right after ``session.updated`` comes back.
        It is fire-and-forget: the gateway sends no ack.
        """
        await super().configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            modality=modality,
            audio_format=audio_format,
        )
        await self._send_nur_configure()

    @staticmethod
    def _nur_configure_payload() -> Optional[Dict[str, Any]]:
        """Parse ``EESI_NUR_CONFIGURE_JSON``; None when unset or unusable."""
        raw = os.environ.get("EESI_NUR_CONFIGURE_JSON")
        if not raw or not raw.strip():
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(
                f"EESI_NUR_CONFIGURE_JSON is not valid JSON, not sending "
                f"eesi.nur.configure: {e}"
            )
            return None
        if not isinstance(payload, dict):
            logger.warning(
                "EESI_NUR_CONFIGURE_JSON must be a JSON object, not sending "
                f"eesi.nur.configure (got {type(payload).__name__})"
            )
            return None
        for key in _BROWSER_ONLY_NUR_KEYS:
            if payload.get(key):
                logger.warning(
                    f"EESI_NUR_CONFIGURE_JSON: dropping {key}=true, it is "
                    "browser-only and stalls a headless realtime client"
                )
                payload.pop(key)
        return payload

    async def _send_nur_configure(self) -> None:
        """Send ``eesi.nur.configure`` from ``EESI_NUR_CONFIGURE_JSON``, if set."""
        payload = self._nur_configure_payload()
        if payload is None:
            return
        frame = {**payload, "type": "eesi.nur.configure"}
        await self.ws.send(json.dumps(frame))
        logger.info(
            f"EESI Realtime API: sent eesi.nur.configure "
            f"{json.dumps(payload, sort_keys=True)} (session_id={self.session_id})"
        )
