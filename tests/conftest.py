from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiobooks.config import Settings
from audiobooks.llm.fake import FakeLLMProvider
from audiobooks.llm.schemas import CharacterObservation
from audiobooks.services.context import AppContext
from audiobooks.storage.db import Database
from audiobooks.tts.fake import FakeTTSBackend
from audiobooks.voices.library import VoiceLibrary

SAMPLE_RU = """Глава 1

Анна подошла к окну. За стеклом медленно падал снег.

— Ты всё-таки пришёл? — тихо спросила Анна.

Иван снял пальто и посмотрел на неё.

— Я обещал, — ответил Иван с лёгкой улыбкой.

Глава 2

Утром снег перестал.

— Пойдём гулять? — спросила Анна.

— Пойдём, — сказал Иван.
"""


def make_voice_library(root: Path) -> Path:
    """A small library: narrator + 2 male + 2 female reference voices (silent WAVs)."""
    meta = root / "metadata"
    refs = root / "references"
    meta.mkdir(parents=True)
    refs.mkdir()
    voices = [
        ("narrator_01", "unknown", ["narrator", "character"]),
        ("male_01", "male", ["character"]),
        ("male_02", "male", ["character"]),
        ("female_01", "female", ["character"]),
        ("female_02", "female", ["character"]),
    ]
    for vid, gender, roles in voices:
        sf.write(refs / f"{vid}.wav", np.zeros(2400, dtype=np.float32), 24000)
        (meta / f"{vid}.json").write_text(
            json.dumps(
                {
                    "id": vid,
                    "name": vid,
                    "gender": gender,
                    "age_group": "adult",
                    "roles": roles,
                    "reference_audio": f"references/{vid}.wav",
                    "reference_text": "test",
                }
            ),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        voice_library_dir=make_voice_library(tmp_path / "voices"),
        tts_backend="fake",
        narrator_voice="narrator_01",
        llm_auto_manage=False,
        audio_loudnorm=False,
    )


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider(
        characters=[
            CharacterObservation(id="anna", name="Анна", gender="female", age_group="adult"),
            CharacterObservation(id="ivan", name="Иван", gender="male", age_group="adult"),
        ]
    )


@pytest.fixture
def fake_tts() -> FakeTTSBackend:
    return FakeTTSBackend()


@pytest.fixture
def ctx(settings: Settings, fake_llm: FakeLLMProvider, fake_tts: FakeTTSBackend) -> AppContext:
    return AppContext(
        settings=settings,
        db=Database(settings.db_path),
        library=VoiceLibrary(settings.voice_library_dir),
        llm_factory=lambda _s: fake_llm,
        tts_factory=lambda _s, _l: fake_tts,
        manage_llm_server=False,
    )


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE_RU, encoding="utf-8")
    return path
