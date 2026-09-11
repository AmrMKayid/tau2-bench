"""Discrete-time adapter for EESI's realtime API.

EESI speaks the GA OpenAI Realtime protocol (see ``eesi/provider.py``), so the
tick loop, event handling and barge-in truncation are OpenAI's verbatim — this
subclass only swaps in the EESI provider. Duplicating
``_execute_tick`` / ``_process_event`` here would mean two copies of the same
protocol that silently drift apart; if EESI's events ever diverge, override the
one method that diverges rather than forking the file.
"""

from typing import Optional

from tau2.data_model.audio import AudioFormat
from tau2.voice.audio_native.eesi.provider import EesiRealtimeProvider
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)


class DiscreteTimeEesiAdapter(DiscreteTimeOpenAIAdapter):
    """Adapter for discrete-time simulation against an EESI realtime endpoint."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[EesiRealtimeProvider] = None,
        audio_format: Optional[AudioFormat] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
            model=model,
            reasoning_effort=reasoning_effort,
            provider=provider,
            audio_format=audio_format,
        )
        self.base_url = base_url

    @property
    def provider(self) -> EesiRealtimeProvider:
        if self._provider is None:
            self._provider = EesiRealtimeProvider(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                base_url=self.base_url,
            )
        return self._provider
