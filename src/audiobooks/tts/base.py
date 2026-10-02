"""TTS backend interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from audiobooks.models.script import Emotion
from audiobooks.models.voice import Voice


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    voice: Voice
    language: str  # TTS language name, e.g. "Russian"
    emotion: Emotion = Emotion.NEUTRAL
    seed: int = 0


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # float32 mono, -1..1
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)


class TTSBackend(Protocol):
    """A speech generator. ``close`` must release all model/GPU memory."""

    name: str

    def model_tag(self, voice: Voice) -> str:
        """Identifier of the model that would voice ``voice`` (part of the cache key)."""
        ...

    def synthesize(self, request: SynthesisRequest) -> AudioClip: ...

    def close(self) -> None: ...


LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
}


def tts_language(setting: str, book_language: str) -> str:
    if setting and setting.lower() != "auto":
        return setting
    code = (book_language or "").split("-")[0].lower()
    return LANGUAGE_NAMES.get(code, "Auto")


EMOTION_INSTRUCTIONS = {
    Emotion.CALM: "Speak calmly and softly.",
    Emotion.HAPPY: "Speak in a happy, warm tone.",
    Emotion.SAD: "Speak in a sad, subdued tone.",
    Emotion.ANGRY: "Speak in an angry, tense tone.",
    Emotion.AFRAID: "Speak in a frightened, nervous tone.",
    Emotion.WHISPER: "Speak quietly, almost whispering.",
    Emotion.EXCITED: "Speak in an excited, energetic tone.",
    Emotion.SERIOUS: "Speak in a serious, firm tone.",
}
