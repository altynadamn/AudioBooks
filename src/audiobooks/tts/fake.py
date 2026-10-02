"""Tone generator used in tests and dry runs. Produces no speech."""

from __future__ import annotations

import hashlib

import numpy as np

from audiobooks.models.voice import Voice
from audiobooks.tts.base import AudioClip, SynthesisRequest


class FakeTTSBackend:
    name = "fake"
    sample_rate = 24_000

    def __init__(self) -> None:
        self.calls: list[SynthesisRequest] = []
        self.closed = False

    def model_tag(self, voice: Voice) -> str:
        return "fake-tone"

    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        self.calls.append(request)
        # a distinct pitch per voice makes casting audible in dry runs
        h = int(hashlib.sha256(request.voice.id.encode()).hexdigest()[:4], 16)
        freq = 180 + (h % 240)
        duration = min(8.0, 0.25 + 0.03 * len(request.text))
        t = np.arange(int(duration * self.sample_rate), dtype=np.float32) / self.sample_rate
        samples = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        return AudioClip(samples=samples, sample_rate=self.sample_rate)

    def close(self) -> None:
        self.closed = True
