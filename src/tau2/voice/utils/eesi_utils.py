"""Text-to-speech through EESI's OpenAI-compatible /v1/audio/speech.

Exists so a voice run can be driven end to end on EESI's own stack, without an
ElevenLabs account. It is **not** the official tau-voice user simulator: the
personas are Sierra's, designed around specific ElevenLabs voices, and a run
that synthesises them with different voices is not comparable to a leaderboard
number. Use it to develop and to get a directional read; use ElevenLabs (or ask
Sierra to run the evaluation) for anything submitted.

The audio effects that follow synthesis -- background noise, burst noise,
telephony band-limiting, frame drops -- are tau-bench's own and are unaffected
by this choice. Only the voice identity changes.
"""

import io
import os
import wave
from typing import Optional

import requests
from dotenv import load_dotenv
from loguru import logger

from tau2.config import DEFAULT_EESI_API_URL, DEFAULT_EESI_TTS_MODEL
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.voice.utils.elevenlabs_utils import AUDIO_TAG_PATTERN, PAUSE_TAG_PATTERN

load_dotenv()

#: WAV, not PCM: the response is self-describing, so the sample rate comes from
#: the file rather than from an assumption about the serving backend.
RESPONSE_FORMAT = "wav"
REQUEST_TIMEOUT_SECONDS = 120

#: A turn the engine has nothing to say for -- ``[pause]``, a lone ``...``, a
#: backchannel the model renders as no speech -- comes back as a WAV with zero
#: frames. The streaming user simulator refuses a message with no audio and the
#: whole simulation is retried from scratch (five minutes of a capped task,
#: three times over). Silence is what that turn meant, so hand back silence:
#: at least this long, and a little longer for longer text.
SILENCE_MIN_SECONDS = 0.4
SILENCE_SECONDS_PER_CHAR = 0.05
SILENCE_MAX_SECONDS = 1.5
SILENCE_SAMPLE_RATE = 24_000


def silence_for(text: str, sample_rate: int = SILENCE_SAMPLE_RATE) -> AudioData:
    """PCM_S16LE mono silence standing in for a turn that produced no speech."""
    seconds = min(
        SILENCE_MAX_SECONDS,
        max(SILENCE_MIN_SECONDS, len(text.strip()) * SILENCE_SECONDS_PER_CHAR),
    )
    frames = b"\x00\x00" * int(seconds * sample_rate)
    return AudioData(
        data=frames,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate, channels=1
        ),
    )


def _strip_elevenlabs_tags(text: str) -> str:
    """Remove ``[cough]``-style tags, which only ElevenLabs v3 understands.

    Every other engine reads them out loud. The vocal-tic effects that insert
    them are an ElevenLabs feature; dropping them here means a tic becomes a
    plain utterance rather than the word "cough".
    """
    cleaned = AUDIO_TAG_PATTERN.sub("", PAUSE_TAG_PATTERN.sub("...", text))
    return " ".join(cleaned.split())


def tts_eesi(
    text: str,
    config: ElevenLabsTTSConfig,
    base_url: Optional[str] = None,
) -> AudioData:
    """Synthesize *text* with EESI TTS.

    Args:
        text: The text to synthesize.
        config: Shared TTS config; ``voice_id`` is the EESI voice
            (``ev_1a2b3c4d``, from ``GET /v1/voices``) and ``model_id`` the
            EESI model id.
        base_url: API root. Defaults to ``EESI_API_URL`` then the dev
            deployment.

    Returns:
        AudioData carrying PCM_S16LE at whatever rate the backend returned.

    Raises:
        ValueError: If no API key or voice is configured, or the response is
            empty or not a WAV.
    """
    api_key = config.api_key or os.getenv("EESI_API_KEY")
    if not api_key:
        raise ValueError("EESI_API_KEY not found in config or environment")

    voice_id = config.voice_id
    if not voice_id:
        raise ValueError(
            "No EESI voice configured. Set TAU2_VOICE_ID_<PERSONA> to a voice id "
            "from GET /v1/voices (e.g. TAU2_VOICE_ID_MATT_DELANEY=ev_1a2b3c4d)."
        )

    url = (base_url or os.getenv("EESI_API_URL") or DEFAULT_EESI_API_URL).rstrip("/")
    model = config.model_id or DEFAULT_EESI_TTS_MODEL

    spoken = _strip_elevenlabs_tags(text)
    if not spoken.strip(". "):
        logger.debug(f"EESI TTS: nothing to say for '{text}', returning silence")
        return silence_for(text)

    payload = {
        "model": model,
        "input": spoken,
        "voice": voice_id,
        "response_format": RESPONSE_FORMAT,
    }
    text_preview = text[:50] + "..." if len(text) > 50 else text
    logger.debug(f"EESI TTS: '{text_preview}' (voice={voice_id}, model={model})")

    response = requests.post(
        f"{url}/audio/speech",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        # The OpenAI error envelope carries the actionable half (unknown voice,
        # quota, unsupported format); the status alone does not.
        raise ValueError(
            f"EESI TTS failed ({response.status_code}): {response.text[:500]}"
        )
    if not response.content:
        raise ValueError(f"EESI TTS returned empty audio for text: '{text}'")

    try:
        with wave.open(io.BytesIO(response.content)) as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except wave.Error as exc:
        raise ValueError(f"EESI TTS returned a non-WAV response: {exc}") from exc

    if sample_width != 2:
        raise ValueError(
            f"EESI TTS returned {sample_width * 8}-bit audio; expected 16-bit PCM"
        )

    if not frames:
        logger.warning(
            f"EESI TTS returned no audio for '{text_preview}', substituting silence"
        )
        return silence_for(text, sample_rate=sample_rate)

    logger.debug(f"EESI TTS: {len(frames)} bytes at {sample_rate} Hz")
    return AudioData(
        data=frames,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
            channels=channels,
        ),
    )
