"""EESI realtime provider (api.eesi.ai / nur-realtime-v1).

EESI's ``/v1/realtime`` speaks the GA OpenAI Realtime protocol: the same
``session.update`` shape, the same ``input_audio_buffer.append``, the same
``response.output_audio.delta`` / ``response.output_audio_transcript.delta`` /
``response.function_call_arguments.done`` server events, and the same
``function_call_output`` + ``response.create`` tool-result handshake. So this
provider is the OpenAI one pointed at a different host, not a reimplementation
of it — anything that drifts in the OpenAI provider should drift here too.

Two things differ, and both live in ``connect``:

- **The endpoint is configurable.** EESI runs a dev deployment, a production
  one and a bare speech orchestrator (the ``s2s`` pool, no gateway in front),
  and the same benchmark has to be runnable against each. ``EESI_REALTIME_URL``
  selects one; the default is dev.
- **The greeting is not the first frame.** EESI's gateway sends its own
  declarations before it relays anything from the speech server: the EU AI Act
  Art. 50(2) synthetic-speech marker, the recording disclosure, and the
  ``eesi.session`` handle used for resume. ``session.created`` arrives *after*
  them, so connect scans for it instead of asserting on frame one.
"""

import json
import os
from typing import Optional

import websockets
from loguru import logger

from tau2.config import (
    DEFAULT_EESI_REALTIME_BASE_URL,
    DEFAULT_EESI_REALTIME_MODEL,
)
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider

#: Gateway frames that may precede ``session.created``. Scanning is bounded so a
#: misconfigured endpoint fails at connect rather than hanging the simulation.
MAX_GREETING_FRAMES = 16


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
        self.session_id = None

    @websocket_retry
    async def connect(self) -> None:
        """Open the socket and wait for ``session.created``.

        Unlike OpenAI's endpoint, EESI's gateway speaks first: it emits the
        synthetic-speech marker, the recording disclosure and the
        ``eesi.session`` resume handle before relaying the speech server's
        ``session.created``. Those frames are logged and skipped.

        Raises:
            RuntimeError: If no ``session.created`` arrives within
                ``MAX_GREETING_FRAMES``, or the endpoint reports an error.
        """
        if self.is_connected:
            return

        url = f"{self.base_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        self.ws = await websockets.connect(url, additional_headers=headers)

        for _ in range(MAX_GREETING_FRAMES):
            data = json.loads(await self.ws.recv())
            event_type = data.get("type")

            if event_type == "session.created":
                session_data = data.get("session", {})
                self.session_id = session_data.get("id")
                logger.info(
                    f"EESI Realtime API: session created "
                    f"(url={self.base_url}, model={self.model}, "
                    f"session_id={self.session_id})"
                )
                return

            if event_type == "error":
                message = data.get("error", {}).get("message", "unknown error")
                raise RuntimeError(f"EESI realtime refused the session: {message}")

            # `eesi.session` carries the resume handle; the marker and the
            # recording disclosure carry compliance state. None of them change
            # how the benchmark drives the socket, so note and move on.
            logger.debug(f"EESI Realtime API: greeting frame {event_type}")

        raise RuntimeError(
            f"No session.created within {MAX_GREETING_FRAMES} frames from {self.base_url}"
        )
