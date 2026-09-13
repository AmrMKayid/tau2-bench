"""EESI TTS: a turn with no speech in it becomes silence, not a crashed run."""

import io
import wave
from unittest import mock

import pytest

from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.voice.utils import eesi_utils
from tau2.voice.utils.eesi_utils import silence_for, tts_eesi


def _wav(frames: bytes, rate: int = 24_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def config() -> ElevenLabsTTSConfig:
    return ElevenLabsTTSConfig(api_key="sk-test", voice_id="ev_test", model_id="m")


def test_silence_scales_with_text_within_bounds() -> None:
    assert silence_for("").duration == pytest.approx(0.4, abs=0.01)
    assert silence_for("uh-huh").duration == pytest.approx(0.4, abs=0.01)
    assert silence_for("x" * 20).duration == pytest.approx(1.0, abs=0.01)
    assert silence_for("x" * 200).duration == pytest.approx(1.5, abs=0.01)
    audio = silence_for("hello", sample_rate=16_000)
    assert audio.format.sample_rate == 16_000
    assert set(audio.data) == {0}


def test_pause_tag_never_reaches_the_backend(config: ElevenLabsTTSConfig) -> None:
    with mock.patch.object(eesi_utils.requests, "post") as post:
        audio = tts_eesi("[pause]", config)
    post.assert_not_called()
    assert audio.format.is_pcm16
    assert audio.duration >= 0.4


def test_empty_wav_from_backend_becomes_silence(config: ElevenLabsTTSConfig) -> None:
    response = mock.Mock(status_code=200, content=_wav(b"", rate=22_050))
    with mock.patch.object(eesi_utils.requests, "post", return_value=response):
        audio = tts_eesi("uh-huh", config)
    assert audio.format.sample_rate == 22_050
    assert audio.duration == pytest.approx(0.4, abs=0.01)


def test_real_audio_passes_through(config: ElevenLabsTTSConfig) -> None:
    frames = b"\x01\x00" * 2400
    response = mock.Mock(status_code=200, content=_wav(frames))
    with mock.patch.object(eesi_utils.requests, "post", return_value=response):
        audio = tts_eesi("Hi, I need to change my flight.", config)
    assert audio.data == frames
    assert audio.format.sample_rate == 24_000
