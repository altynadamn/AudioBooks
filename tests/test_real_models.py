"""Optional smoke tests against the real local models.

Skipped unless AUDIOBOOKS_REAL_MODELS=1. They use your .env (llama-server URL/paths,
Qwen3-TTS model, narrator voice). Run them with the GPU free:

    $env:AUDIOBOOKS_REAL_MODELS = "1"; pytest tests/test_real_models.py -s
"""

from __future__ import annotations

import os

import pytest

from audiobooks.config import Settings
from audiobooks.llm.schemas import ChunkRequest, SpanInput
from audiobooks.voices.library import VoiceLibrary

pytestmark = [
    pytest.mark.real_models,
    pytest.mark.skipif(
        os.environ.get("AUDIOBOOKS_REAL_MODELS") != "1", reason="set AUDIOBOOKS_REAL_MODELS=1"
    ),
]


def test_llm_labels_dialogue() -> None:
    from pathlib import Path

    from audiobooks.llm.openai_provider import OpenAICompatibleProvider
    from audiobooks.llm.server import LlamaServerManager

    settings = Settings()
    request = ChunkRequest(
        book_title="Test",
        chapter_title="1",
        language="ru",
        spans=[
            SpanInput(0, "narration", "Анна подошла к окну."),
            SpanInput(1, "dialogue", "Ты всё-таки пришёл?"),
            SpanInput(2, "narration", "тихо спросила она."),
        ],
    )
    manager = LlamaServerManager(settings, log_dir=Path(settings.data_dir) / "logs")
    if settings.llm_auto_manage:
        manager.start()
    try:
        analysis = OpenAICompatibleProvider(settings).analyze_chunk(request)
    finally:
        manager.stop()
    labels = {s.id: s for s in analysis.segments}
    assert labels[1].type == "dialogue"
    assert labels[2].type == "narration"
    assert any(c.name.startswith("Анн") for c in analysis.characters)


def test_qwen_tts_generates_audio() -> None:
    from audiobooks.tts.base import SynthesisRequest
    from audiobooks.tts.qwen import QwenTTSBackend

    settings = Settings()
    library = VoiceLibrary(settings.voice_library_dir)
    backend = QwenTTSBackend(settings, library)
    try:
        clip = backend.synthesize(
            SynthesisRequest("Проверка связи.", library.get(settings.narrator_voice), "Russian")
        )
    finally:
        backend.close()
    assert clip.duration > 0.3
