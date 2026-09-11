"""EESI realtime API integration for audio-native voice processing.

EESI's ``/v1/realtime`` is OpenAI-Realtime-compatible, so this package reuses
the OpenAI events, VAD config and tick loop and changes only the endpoint and
the connect handshake.

Reference: https://docs.eesi.ai
"""

from tau2.voice.audio_native.eesi.discrete_time_adapter import DiscreteTimeEesiAdapter
from tau2.voice.audio_native.eesi.provider import EesiRealtimeProvider

__all__ = [
    "DiscreteTimeEesiAdapter",
    "EesiRealtimeProvider",
]
