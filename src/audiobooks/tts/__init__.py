"""Text-to-speech backends."""

from __future__ import annotations

from audiobooks.config import Settings
from audiobooks.tts.base import TTSBackend
from audiobooks.voices.library import VoiceLibrary


def create_backend(settings: Settings, library: VoiceLibrary) -> TTSBackend:
    if settings.tts_backend == "fake":
        from audiobooks.tts.fake import FakeTTSBackend

        return FakeTTSBackend()
    from audiobooks.tts.qwen import QwenTTSBackend

    return QwenTTSBackend(settings, library)
